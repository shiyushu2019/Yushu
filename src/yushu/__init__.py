from importlib.metadata import PackageNotFoundError, version
from . import functional
from .functional import (
    HelloWorld,
)

try:
    __version__ = version("yushu")     # 从安装元数据读，不用手改
except PackageNotFoundError:           # 直接在源码目录里跑时
    __version__ = "0.0.0"

__all__ = ["functional", "HelloWorld"]