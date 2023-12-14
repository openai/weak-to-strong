import fire
import numpy as np
import torch
import tqdm
from data import get_imagenet
from models import alexnet, resnet50_dino, vitb8_dino
from torch import nn


def get_model(name):
    if name == "alexnet":
        model = alexnet()
    elif name == "resnet50_dino":
        model = resnet50_dino()
    elif name == "vitb8_dino":
        model = vitb8_dino()
    else:
        raise ValueError(f"Unknown model {name}")
    model.cuda()
    model.eval()
    model = nn.DataParallel(model)
    return model


def get_embeddings(model, loader):
    all_embeddings, all_y, all_probs = [], [], []

    for x, y in tqdm.tqdm(loader):
        output = model(x.cuda())
        if len(output) == 2:
            embeddings, logits = output
            probs = torch.nn.functional.softmax(logits, dim=-1).detach().cpu()
            all_probs.append(probs)
        else:
            embeddings = output

        all_embeddings.append(embeddings.detach().cpu())
        all_y.append(y)

    all_embeddings = torch.cat(all_embeddings, axis=0)
    all_y = torch.cat(all_y, axis=0)
    if len(all_probs) > 0:
        all_probs = torch.cat(all_probs, axis=0)
        acc = (torch.argmax(all_probs, dim=1) == all_y).float().mean()
    else:
        all_probs = None
        acc = None
    return all_embeddings, all_y, all_probs, acc


def train_logreg(
    x_train,
    y_train,
    eval_datasets,
    n_epochs=10,
    weight_decay=0.0,
    lr=1.0e-3,
    batch_size=100,
    n_classes=1000,
):
    x_train = x_train.float()
    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size)

    d = x_train.shape[1]
    model = torch.nn.Linear(d, n_classes).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs)

    results = {f"{key}_all": [] for key in eval_datasets.keys()}
    for epoch in (pbar := tqdm.tqdm(range(n_epochs), desc="Epoch 0")):
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            schedule.step()
            if len(y.shape) > 1:
                y = torch.argmax(y, dim=1)
            correct += (torch.argmax(pred, -1) == y).detach().float().sum().item()
            total += len(y)
        pbar.set_description(f"Epoch {epoch}, Train Acc {correct / total:.3f}")

        for key, (x_test, y_test) in eval_datasets.items():
            x_test = x_test.float().cuda()
            pred = torch.argmax(model(x_test), axis=-1).detach().cpu()
            acc = (pred == y_test).float().mean()
            results[f"{key}_all"].append(acc)

    for key in eval_datasets.keys():
        results[key] = results[f"{key}_all"][-1]
    return results


def main(
    batch_size: int = 128,
    weak_model_name: str = "alexnet",
    strong_model_name: str = "resnet50_dino",
    n_train: int = 40000,
    seed: int = 0,
    data_path: str = "/root/",
    n_epochs: int = 10,
    lr: float = 1e-3,
):
    weak_model = get_model(weak_model_name)
    strong_model = get_model(strong_model_name)
    _, loader = get_imagenet(data_path, split="val", batch_size=batch_size, shuffle=False)
    print("Getting weak labels...")
    _, gt_labels, weak_labels, weak_acc = get_embeddings(weak_model, loader)
    print(f"Weak label accuracy: {weak_acc:.3f}")
    print("Getting strong embeddings...")
    embeddings, strong_gt_labels, _, _ = get_embeddings(strong_model, loader)
    assert torch.all(gt_labels == strong_gt_labels)
    del strong_gt_labels

    order = np.arange(len(embeddings))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    x = embeddings[order]
    y = gt_labels[order]
    yw = weak_labels[order]
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    yw_train, yw_test = yw[:n_train], yw[n_train:]
    yw_test = torch.argmax(yw_test, dim=1)
    eval_datasets = {"test": (x_test, y_test), "test_weak": (x_test, yw_test)}

    print("Training logreg on weak labels...")
    results_weak = train_logreg(x_train, yw_train, eval_datasets, n_epochs=n_epochs, lr=lr)
    print(f"Final accuracy: {results_weak['test']:.3f}")
    print(f"Final supervisor-student agreement: {results_weak['test_weak']:.3f}")
    print(f"Accuracy by epoch: {[acc.item() for acc in results_weak['test_all']]}")
    print(
        f"Supervisor-student agreement by epoch: {[acc.item() for acc in results_weak['test_weak_all']]}"
    )

    print("Training logreg on ground truth labels...")
    results_gt = train_logreg(x_train, y_train, eval_datasets, n_epochs=n_epochs, lr=lr)
    print(f"Final accuracy: {results_gt['test']:.3f}")
    print(f"Accuracy by epoch: {[acc.item() for acc in results_gt['test_all']]}")

    print("\n\n" + "=" * 100)
    print(f"Weak label accuracy: {weak_acc:.3f}")
    print(f"Weak→Strong accuracy: {results_weak['test']:.3f}")
    print(f"Strong accuracy: {results_gt['test']:.3f}")
    print("=" * 100)


if __name__ == "__main__":
    fire.Fire(main)
