import os
import subprocess
import sys
from typing import List, Union

import fire


def main(
    model_sizes: Union[List[str], str], weak_epochs: int, strong_epochs: int, **kwargs
):
    if isinstance(model_sizes, str):
        model_sizes = model_sizes.split(",")
    assert (
        "weak_model_size" not in kwargs
        and "model_size" not in kwargs
        and "weak_labels_path" not in kwargs
    ), "Need to use model_sizes when using sweep.py"
    basic_args = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "train_simple.py"),
    ]
    for key, value in kwargs.items():
        if key != "eval_every":
            basic_args.extend([f"--{key}", str(value)])

    print("Running ground truth models")
    for model_size in model_sizes:
        subprocess.run(
            basic_args + ["--model_size", model_size, "--epochs", str(weak_epochs)],
            check=True,
        )

    if "eval_every" in kwargs:
        basic_args.extend(["--eval_every", str(kwargs["eval_every"])])

    print("Running transfer models")
    for i in range(len(model_sizes)):
        for j in range(i, len(model_sizes)):
            weak_model_size = model_sizes[i]
            strong_model_size = model_sizes[j]
            print(f"Running weak {weak_model_size} to strong {strong_model_size}")
            subprocess.run(
                basic_args
                + [
                    "--weak_model_size",
                    weak_model_size,
                    "--model_size",
                    strong_model_size,
                    "--epochs",
                    str(strong_epochs),
                ],
                check=True,
            )


if __name__ == "__main__":
    # see train_simple.py for valid args
    fire.Fire(main)
