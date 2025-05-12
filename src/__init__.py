"""
Chat2Table - 一个用于将图片和PDF转换为结构化数据的工具包
"""

__version__ = "0.1.0"

from .models.config import ModelConfig
from .chains.image_processing import ImageProcessingChain

__all__ = ["ModelConfig", "ImageProcessingChain"] 