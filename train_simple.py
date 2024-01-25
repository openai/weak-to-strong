import json
import os
import random
import subprocess
from typing import Optional

import fire
import numpy as np
from datasets import load_from_disk
import wandb

import weak_to_strong.logger as logger
from weak_to_strong.common import get_tokenizer
from weak_to_strong.config import MODELS_DICT, get_config_foldername, loss_dict
from weak_to_strong.datasets import (
    VALID_DATASETS,
    tokenize_dataset,
    load_and_process_dataset,
)
from weak_to_strong.train import train_and_save_model


def main(
    batch_size: int = 32,
    max_ctx: int = 1024,
    ds_name: str = "sciq",
    loss: str = "xent",
    n_train1_docs: int = 20000,
    n_train2_docs: int = 10000,
    n_test_docs: int = 10000,
    model_size: str = "gpt2",
    lr: Optional[float] = None,
    optim: Optional[str] = None,
    weak_epochs: int = 1,
    strong_epochs: int = 1,
    force_retrain: bool = False,
    seed: int = 0,
    minibatch_size_per_device: Optional[int] = None,
    train_with_dropout: bool = False,
    results_folder: str = "/tmp/results",
    linear_probe: bool = False,
    lr_schedule: str = "cosine_anneal",
    # Note: you can pass either weak_model_size or weak_labels_path. If you pass
    # weak_model_size, we will guess the path to the weak labels based on the weak
    # model. If you pass weak_labels_path, we will use that path instead.
    # If you pass neither, we will train on ground truth.
    weak_model_size: Optional[str] = None,
    weak_labels_path: Optional[str] = None,
    sweep_subfolder: str = "default",
    # Set to a very large value so that by default we don't do any intermediate evals but
    # still do final evals (which requires eval_every to be set to a non-zero, non-None value)
    eval_every: int = 10000000,
    sync_command: Optional[str] = None,
):  
    assert (
        ds_name in VALID_DATASETS
    ), f"Unknown dataset {ds_name} not in {VALID_DATASETS}"
    assert (
        weak_model_size is None or weak_labels_path is None
    ), "Can't pass both weak_model_size and weak_labels_path"
    model_config = MODELS_DICT[model_size]

    # only evaluate intermediately for the student
    eval_every = eval_every if weak_labels_path is not None else 10000000
    epochs = strong_epochs if weak_labels_path is not None else weak_epochs

    # this is per device!
    if minibatch_size_per_device is None:
        minibatch_size_per_device = model_config.minibatch_size_per_device or 1

    use_default_lr = False
    if lr is None:
        assert batch_size == 32, (
            "Learning rates were tuned on batch size 32, you probably want to sweep LR "
            "if you are tuning batch size"
        )
        lr = model_config.default_lr
        use_default_lr = True

    if optim is None:
        optim = model_config.default_optimizer

    # The commented out terms are the ones that should not change final results
    config = {
        "batch_size": batch_size,
        "max_ctx": max_ctx,
        "ds_name": ds_name,
        "loss": loss,
        "n_train1_docs": n_train1_docs,
        "n_train2_docs": n_train2_docs,
        "n_test_docs": n_test_docs,
        "model_size": model_size,
        "lr": lr,
        "optim": optim,
        ("strong_epochs" if weak_labels_path is not None else "weak_epochs"): epochs,
        # "force_retrain": force_retrain,
        "seed": seed,
        # "minibatch_size_per_device": minibatch_size_per_device,
        "train_with_dropout": train_with_dropout,
        # "results_folder": results_folder,
        "linear_probe": linear_probe,
        "lr_schedule": lr_schedule,
        "eval_every": eval_every,
        # "sweep_subfolder": sweep_subfolder,
    }

    if weak_model_size is not None:
        weak_model_config = config.copy()
        weak_model_config["model_size"] = weak_model_size
        weak_model_config["loss"] = "xent"
        if use_default_lr:
            weak_model_config["lr"] = MODELS_DICT[weak_model_size].default_lr

        weak_model_config_name = get_config_foldername(weak_model_config)

        weak_labels_path = (
            results_folder
            + "/"
            + sweep_subfolder
            + "/"
            + weak_model_config_name
            + "/weak_labels"
        )

    eval_batch_size = model_config.eval_batch_size
    random.seed(seed)

    print("DS NAME:", ds_name)
    # Load dataset
    dataset = load_and_process_dataset(
        ds_name,
        seed=seed,
        split_sizes=dict(train=n_train1_docs + n_train2_docs, test=n_test_docs),
    )

    # Split the training dataset in half
    train_dataset, test_ds = dataset["train"], dataset["test"]  # type: ignore

    if weak_labels_path is None:  # train on ground truth
        # split off half for getting weak labels
        split_data = train_dataset.train_test_split(test_size=n_train2_docs)
        train1_ds, train2_ds = split_data["train"], split_data["test"]
        print("len(train1):", len(train1_ds), "len(train2):", len(train2_ds))
        config_name = get_config_foldername(config)
    else:
        if not weak_labels_path.endswith("weak_labels"):
            weak_labels_path = weak_labels_path + "/weak_labels"
        if sync_command is not None:
            sync_command_list = sync_command.split(" ")
            sync_command_list.extend(
                [
                    "download",
                    weak_labels_path.replace("/weak_labels", ""),
                    results_folder,
                ]
            )
            print(f"Running sync command: {' '.join(sync_command_list)}")
            result = subprocess.run(sync_command_list, check=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Sync command failed with return code {result.returncode}"
                )
        train1_ds = load_from_disk(weak_labels_path)
        train2_ds = None

        weak_model_config = json.load(
            open(weak_labels_path.replace("weak_labels", "config.json"))
        )
        config["weak_model_size"] = weak_model_config["model_size"]
        config_name = get_config_foldername(config)
        config["weak_model"] = weak_model_config

    save_path = os.path.join(results_folder, sweep_subfolder, config_name)
    logger.configure(
        name="{sweep_subfolder}_{config_name}_{datetime_now}",
        save_path=save_path,
        sweep_subfolder=sweep_subfolder,
        config_name=config_name,
    )
    wandb.init(
        project="weak-to-strong",
        config=config,
        group=sweep_subfolder,
        job_type="gt" if weak_labels_path is None else "w2s",
        name=f"{model_size.split('/')[-1]}_{ds_name}_{loss}",
        dir=results_folder,
        reinit=True,
    )

    # Tokenize datasets
    tokenizer = get_tokenizer(model_config.name)
    train1_ds = tokenize_dataset(train1_ds, tokenizer, max_ctx)  # type: ignore
    test_ds = tokenize_dataset(test_ds, tokenizer, max_ctx)  # type: ignore
    if train2_ds:
        train2_ds = tokenize_dataset(train2_ds, tokenizer, max_ctx)

    loss_fn = loss_dict[loss]
    print(f"Training model {model_size}")
    test_results, weak_ds = train_and_save_model(
        model_config,
        train1_ds,  # this has weak labels iff weak_labels_path is not None
        test_ds,  # this has ground truth labels no matter what
        inference_ds=train2_ds,  # make weak training dataset for strong model
        batch_size=batch_size,
        save_path=save_path,
        loss_fn=loss_fn,
        lr=lr,
        epochs=epochs,
        force_retrain=force_retrain,
        eval_batch_size=eval_batch_size,
        minibatch_size_per_device=minibatch_size_per_device,
        train_with_dropout=train_with_dropout,
        linear_probe=linear_probe,
        lr_schedule=lr_schedule,
        optimizer_name=optim,
        eval_every=eval_every,
    )

    if weak_ds is not None:
        weak_ds.save_to_disk(save_path + "/" + "weak_labels")

    acc = np.mean([x["acc"] for x in test_results])  # type: ignore
    res_dict = {"accuracy": acc}
    print("accuracy:", acc)

    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(save_path, "results_summary.json"), "w") as f:
        json.dump(res_dict, f, indent=2)

    if sync_command is not None:
        print("Syncing results to remote storage...")
        try:
            sync_command_list = sync_command.split(" ")
            sync_command_list.extend(["upload", save_path, results_folder])
            print(f"Running sync command: {' '.join(sync_command_list)}")
            result = subprocess.run(sync_command_list, check=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Sync command failed with return code {result.returncode}"
                )
        except Exception as e:
            raise RuntimeError("Failed to sync results to remote storage.") from e


if __name__ == "__main__":
    fire.Fire(main)
