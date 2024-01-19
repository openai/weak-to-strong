import gc

import torch
from transformers import AutoTokenizer


def to_batch(x, batch_size: int, start: int = 0, end: int | None = None):
    """Helper function to split a dataset into batches,
    with the last batch being smaller if necessary."""
    end = end if end is not None else len(x)
    for i in range(start, end, batch_size):
        if i >= len(x):
            return
        yield x[i : i + batch_size]


def get_tokenizer(model_name: str):
    """
    This function returns a tokenizer based on the model name.

    Parameters:
    model_name: The name of the model for which the tokenizer is needed.

    Returns:
    A tokenizer for the specified model.
    """
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def clear_mem(verbose: bool = False):
    """
    This function is used to clear the memory allocated by PyTorch.
    It does so by calling the garbage collector to release unused GPU memory.
    After clearing the memory, it prints the current amount of memory still
    allocated by PyTorch (post-clean).

    Parameters:
    verbose (bool): Whether to print additional information.
    """

    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB"
    )

    if verbose:

        def try_attr(x, a):
            try:
                return getattr(x, a)
            except Exception:
                # amazing that this can cause...
                # (AttributeError, OSError, AssertionError, RuntimeError, ModuleNotFoundError)
                return None

        for obj in gc.get_objects():
            if torch.is_tensor(obj) or torch.is_tensor(try_attr(obj, "data")):
                print(type(obj), obj.size(), obj.dtype)
