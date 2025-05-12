#!/usr/bin/env python
"""
Chat2Table - 一个用于将图片和PDF转换为结构化数据的工具
"""

from .models.config import ModelConfig
from .chains.image_processing import ImageProcessingChain

__version__ = "0.1.0"

def get_processor(api_key: str) -> ImageProcessingChain:
    """
    获取图像处理器实例
    
    Args:
        api_key: API密钥
        
    Returns:
        ImageProcessingChain: 图像处理器实例
    """
    config = ModelConfig(api_key=api_key)
    return ImageProcessingChain(config)

# TODO: 添加更多工具函数和类 

