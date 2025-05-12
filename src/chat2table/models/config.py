from typing import Optional
from langchain_openai import ChatOpenAI
import os
from ..utils.logger import logger

class ConfigError(Exception):
    """配置错误"""
    pass

class ModelConfig:
    """模型配置类"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 recognition_model: str = "qwen-vl-max",
                 analysis_model: str = "qwen-max",
                 temperature: float = 0.1,
                 max_tokens: int = 2000):
        """
        初始化模型配置
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            recognition_model: 识别模型名称
            analysis_model: 分析模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            
        Raises:
            ConfigError: 当配置无效时抛出
        """
        self._validate_api_key(api_key)
        self._validate_url(base_url)
        self._validate_models(recognition_model, analysis_model)
        self._validate_parameters(temperature, max_tokens)
        
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url
        self.recognition_model = recognition_model
        self.analysis_model = analysis_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def _validate_api_key(self, api_key: Optional[str]) -> None:
        """验证API密钥"""
        if not api_key and not os.getenv("DASHSCOPE_API_KEY"):
            raise ConfigError("API密钥未设置，请提供api_key参数或设置DASHSCOPE_API_KEY环境变量")
            
    def _validate_url(self, base_url: str) -> None:
        """验证URL"""
        if not base_url.startswith(('http://', 'https://')):
            raise ConfigError(f"无效的base_url: {base_url}")
            
    def _validate_models(self, recognition_model: str, analysis_model: str) -> None:
        """验证模型名称"""
        if not recognition_model or not analysis_model:
            raise ConfigError("模型名称不能为空")
            
    def _validate_parameters(self, temperature: float, max_tokens: int) -> None:
        """验证参数"""
        if not 0 <= temperature <= 1:
            raise ConfigError(f"temperature必须在0到1之间，当前值: {temperature}")
        if max_tokens <= 0:
            raise ConfigError(f"max_tokens必须大于0，当前值: {max_tokens}")
        
    def get_recognition_model(self) -> ChatOpenAI:
        """获取识别模型"""
        try:
            return ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.recognition_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        except Exception as e:
            logger.error(f"创建识别模型失败: {str(e)}")
            raise ConfigError(f"创建识别模型失败: {str(e)}")
        
    def get_analysis_model(self) -> ChatOpenAI:
        """获取分析模型"""
        try:
            return ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.analysis_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            ) 
        except Exception as e:
            logger.error(f"创建分析模型失败: {str(e)}")
            raise ConfigError(f"创建分析模型失败: {str(e)}") 