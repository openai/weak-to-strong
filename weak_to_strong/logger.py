import json
import os
from datetime import datetime

import wandb


def append_to_jsonl(path: str, data: dict):
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


class WandbLogger(object):
    CURRENT = None

    log_path = None

    def __init__(
        self,
        **kwargs,
    ):
        project = os.environ.get("WANDB_PROJECT")
        self.use_wandb = project is not None
        if self.use_wandb:
            wandb.init(
                config=kwargs,
                project=project,
                name=kwargs["name"].format(
                    **kwargs, datetime_now=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                )
                if "name" in kwargs
                else None,
            )
        if "save_path" in kwargs:
            self.log_path = os.path.join(kwargs["save_path"], "log.jsonl")
            if not os.path.exists(kwargs["save_path"]):
                os.makedirs(kwargs["save_path"])
        self._log_dict = {}

    def logkv(self, key, value):
        self._log_dict[key] = value

    def logkvs(self, d):
        self._log_dict.update(d)

    def dumpkvs(self):
        if self.use_wandb:
            wandb.log(self._log_dict)
        if self.log_path is not None:
            append_to_jsonl(self.log_path, self._log_dict)
        self._log_dict = {}

    def shutdown(self):
        if self.use_wandb:
            wandb.finish()


def is_configured():
    return WandbLogger.CURRENT is not None


def get_current():
    assert is_configured(), "WandbLogger is not configured"
    return WandbLogger.CURRENT


def configure(**kwargs):
    if is_configured():
        WandbLogger.CURRENT.shutdown()
    WandbLogger.CURRENT = WandbLogger(**kwargs)
    return WandbLogger.CURRENT


def logkv(key, value):
    assert is_configured(), "WandbLogger is not configured"
    WandbLogger.CURRENT.logkv(key, value)


def logkvs(d):
    assert is_configured(), "WandbLogger is not configured"
    WandbLogger.CURRENT.logkvs(d)


def dumpkvs():
    assert is_configured(), "WandbLogger is not configured"
    WandbLogger.CURRENT.dumpkvs()


def shutdown():
    assert is_configured(), "WandbLogger is not configured"
    WandbLogger.CURRENT.shutdown()
    WandbLogger.CURRENT = None
