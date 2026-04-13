from typing import TypeVar

from hydra.utils import instantiate
from omegaconf import DictConfig

T = TypeVar("T")


def instantiate_assert_type(cfg: DictConfig, expected_type: type[T], *args, **kwargs) -> T:
    obj = instantiate(cfg, *args, **kwargs)
    if cfg.get("_partial_", False):
        obj = obj()
    if isinstance(obj, expected_type):
        return obj
    raise TypeError
