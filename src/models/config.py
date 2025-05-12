from typing import Optional
from langchain_openai import ChatOpenAI
import os

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
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("API密钥未设置，请提供api_key参数或设置DASHSCOPE_API_KEY环境变量")
            
        self.base_url = base_url
        self.recognition_model = recognition_model
        self.analysis_model = analysis_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def get_recognition_model(self) -> ChatOpenAI:
        """获取识别模型"""
        return ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.recognition_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
    def get_analysis_model(self) -> ChatOpenAI:
        """获取分析模型"""
        return ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.analysis_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        ) 