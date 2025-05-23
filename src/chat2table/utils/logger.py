import logging
import os
import sys
from datetime import datetime

class CustomFormatter(logging.Formatter):
    """自定义日志格式化器"""
    def format(self, record):
        # 如果消息是二进制数据，替换为占位符
        if isinstance(record.msg, bytes):
            record.msg = "<binary data>"
        return super().format(record)

def setup_logger(name: str = None) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称，默认为None（根记录器）
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 如果已经有处理器，说明已经配置过，直接返回
    if logger.handlers:
        return logger
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = CustomFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 检查是否是作为独立项目运行
    is_standalone = __package__ is None or __package__ == ""
    
    # 只有在作为独立项目运行时才创建日志文件
    if is_standalone:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = CustomFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# 创建默认的日志记录器
logger = setup_logger("chat2table") 