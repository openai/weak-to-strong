import gc

import torch
from transformers import AutoTokenizer


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
    It does so by calling the garbage collector and emptying the CUDA cache.
    After clearing the memory, it prints the amount of memory allocated by PyTorch.
    """
    gc.collect()
    torch.cuda.empty_cache()
    print(
        "torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
    )

    if verbose:

        def try_attr(x, a):
            try:
                return getattr(x, a)
            except:
                # amazing that this can cause...
                #  (AttributeError, OSError, AssertionError, RuntimeError, ModuleNotFoundError)
                return None

        for obj in gc.get_objects():
            if torch.is_tensor(obj) or torch.is_tensor(try_attr(obj, "data")):
                print(type(obj), obj.size(), obj.dtype)
