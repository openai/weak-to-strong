import os
import sys
from train_simple import main as train_simple_main
from typing import List, Union

import fire


def main(
    model_sizes: Union[List[str], str], train_self_to_self: bool = False, **kwargs
):
    if isinstance(model_sizes, str):
        model_sizes = model_sizes.split(",")
    assert (
        "weak_model_size" not in kwargs
        and "model_size" not in kwargs
        and "weak_labels_path" not in kwargs
    ), "Need to use model_sizes when using sweep.py"

    print("Running ground truth models")
    model_sizes_to_run = model_sizes if train_self_to_self else model_sizes[:-1]
    for model_size in model_sizes_to_run:
        print(f"Running ground truth {model_size}")
        train_simple_main(model_size=model_size, **kwargs)
    
    print("Running transfer models")
    for i in range(len(model_sizes)):
        start = i if train_self_to_self else i + 1
        for j in range(start, len(model_sizes)):
            weak_model_size = model_sizes[i]
            strong_model_size = model_sizes[j]
            print(f"Running weak {weak_model_size} to strong {strong_model_size}")
            train_simple_main(
                model_size=strong_model_size,
                weak_model_size=weak_model_size,
                **kwargs,
            )
    
    print("Finished running all models")


if __name__ == "__main__":
    # see train_simple.py for valid args
    fire.Fire(main)
