import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import logging
import json
from typing import Dict, Any, Optional, Union

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageRecognizer:
    """图像识别器类，用于处理图像识别任务"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                 model_name: str = "qwen-vl-max", debug: bool = True):
        """
        初始化图像识别器
        
        Args:
            api_key: API密钥，如果为None则从环境变量加载
            base_url: API基础URL
            model_name: 使用的模型名称
            debug: 是否启用调试模式
        """
        self.debug = debug
        self.model_name = model_name
        self.base_url = base_url
        
        # 加载环境变量和API密钥
        self._load_env(api_key)
        
        # 设置客户端
        self._setup_client()
        
        # 默认提示词
        self.default_prompt = """请仔细观察并描述图片内容。要求如下：
        1. 详细解释图片中数据或图表所表达的含义，确保内容准确、逻辑清晰。
        2. 不需要进行主观分析，但必须完整、准确地输出图片中包含的所有数据，确保无遗漏。
        """

    def _load_env(self, api_key: Optional[str] = None) -> None:
        """加载环境变量和API密钥"""
        if not load_dotenv():
            logger.warning("无法加载.env文件，将使用提供的API密钥")
        
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("API密钥未设置，请提供api_key参数或设置DASHSCOPE_API_KEY环境变量")

    def _setup_client(self) -> None:
        """配置OpenAI客户端"""
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def _encode_image(self, image_path: str) -> str:
        """
        将图片编码为Base64
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            str: Base64编码的图片数据
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"图片编码失败: {str(e)}")
            raise

    def recognize_image(self, image_path: str, prompt: Optional[str] = None, 
                       stream: bool = False) -> Union[str, Dict[str, Any]]:
        """
        识别图片内容
        
        Args:
            image_path: 图片文件路径
            prompt: 自定义提示词，如果为None则使用默认提示词
            stream: 是否使用流式输出
            
        Returns:
            Union[str, Dict[str, Any]]: 识别结果，如果stream=True则返回字符串，否则返回字典
        """
        try:
            # 检查图片是否存在
            if not Path(image_path).exists():
                raise FileNotFoundError(f"图片文件不存在: {image_path}")
            
            # 编码图片
            base64_image = self._encode_image(image_path)
            
            if self.debug:
                logger.info(f"准备处理图片: {image_path}")
            
            # 构建请求
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': '你是一个专业的图像分析助手，擅长详细描述图像内容。'},
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                            {'type': 'text', 'text': prompt or self.default_prompt}
                        ]
                    }
                ],
                stream=stream
            )
            
            if self.debug:
                logger.info("API请求成功")
            
            # 处理响应
            if stream:
                full_response = ""
                print("\n正在接收响应...")
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        full_response += content
                print("\n")
                return full_response
            else:
                result = response.choices[0].message.content
                if self.debug:
                    logger.info(f"响应内容: {json.dumps(result, ensure_ascii=False)}")
                
                # 返回结构化的结果
                return {
                    "status": "success",
                    "image_path": image_path,
                    "content": result,
                    "model": self.model_name
                }
            
        except Exception as e:
            error_msg = f"处理过程中出现错误: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "image_path": image_path,
                "error": str(e)
            } 