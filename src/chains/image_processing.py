import base64
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import io
from PIL import Image

from ..models.config import ModelConfig
from ..prompts.image_recognition import IMAGE_RECOGNITION_TEMPLATE
from ..prompts.analysis import ANALYSIS_TEMPLATE
from ..utils.logger import logger

class ImageProcessingChain:
    """图像处理链"""
    
    def __init__(self, config: ModelConfig, debug: bool = True):
        """
        初始化图像处理链
        
        Args:
            config: 模型配置
            debug: 是否启用调试模式
        """
        self.config = config
        self.debug = debug
        
        # 初始化模型
        self.recognition_model = config.get_recognition_model()
        self.analysis_model = config.get_analysis_model()
        
        # 初始化输出解析器
        self.output_parser = StrOutputParser()
        
    def _download_image(self, image_url: str) -> bytes:
        """
        从URL下载图片到内存
        
        Args:
            image_url: 图片URL
            
        Returns:
            bytes: 图片二进制数据
        """
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"下载图片失败: {str(e)}")
            raise
            
    def _encode_image(self, image_data: bytes) -> str:
        """
        将图片编码为Base64
        
        Args:
            image_data: 图片二进制数据
            
        Returns:
            str: Base64编码的图片数据
        """
        try:
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            logger.error(f"图片编码失败: {str(e)}")
            raise
            
    def _create_recognition_chain(self):
        """创建识别链"""
        def create_messages(x):
            human_messages = []
            # 添加所有图片
            for img in x["images"]:
                human_messages.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                })
            # 添加文本提示
            human_messages.append({
                "type": "text",
                "text": IMAGE_RECOGNITION_TEMPLATE
            })
            messages = [
                SystemMessage(content="你是一个专业的图像分析助手，擅长详细描述图像内容。"),
                HumanMessage(content=human_messages)
            ]
            # 记录输入提示到日志文件
            logger.debug("=== 图像识别输入提示 ===")
            logger.debug(f"系统提示: {messages[0].content}")
            logger.debug(f"用户提示: {IMAGE_RECOGNITION_TEMPLATE}")
            return messages

        return (
            RunnablePassthrough.assign(
                images=lambda x: [self._encode_image(img) for img in x["image_data_list"]]
            )
            | create_messages
            | self.recognition_model
            | self.output_parser
        )
        
    def _create_analysis_chain(self):
        """创建分析链"""
        def create_messages(x):
            messages = [
                SystemMessage(content="你是一个专业的数据分析助手，擅长从文本中提取结构化信息。"),
                HumanMessage(content=ANALYSIS_TEMPLATE.format(recognition_result=x["recognition_result"]))
            ]
            # 记录输入提示到日志文件
            logger.debug("=== 数据分析输入提示 ===")
            logger.debug(f"系统提示: {messages[0].content}")
            logger.debug(f"用户提示: {ANALYSIS_TEMPLATE}")
            return messages

        return (
            RunnablePassthrough.assign(
                recognition_result=lambda x: x["recognition_result"]
            )
            | create_messages
            | self.analysis_model
            | self.output_parser
        )
        
    def _convert_to_bytes(self, image_data: Union[str, bytes, io.BytesIO, Image.Image]) -> bytes:
        """
        将不同类型的图片数据转换为字节数据
        
        Args:
            image_data: 图片数据，可以是URL、二进制数据、BytesIO对象或PIL.Image对象
            
        Returns:
            bytes: 图片二进制数据
        """
        if isinstance(image_data, str):
            if image_data.startswith(('http://', 'https://', 'file://')):
                return self._download_image(image_data)
            else:
                raise ValueError(f"不支持的URL格式: {image_data}")
        elif isinstance(image_data, io.BytesIO):
            return image_data.getvalue()
        elif isinstance(image_data, bytes):
            return image_data
        elif isinstance(image_data, Image.Image):
            # 将PIL.Image转换为字节数据
            img_byte_arr = io.BytesIO()
            image_data.save(img_byte_arr, format='JPEG')
            return img_byte_arr.getvalue()
        else:
            raise ValueError(f"不支持的图片数据类型: {type(image_data)}")
        
    def process_image(self, image_data: Union[str, bytes, io.BytesIO, Image.Image], stream: bool = False) -> Dict[str, Any]:
        """
        处理单张图片
        
        Args:
            image_data: 图片URL、二进制数据、BytesIO对象或PIL.Image对象
            stream: 是否使用流式输出
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            if self.debug:
                logger.info(f"开始处理图片: {image_data if isinstance(image_data, str) else '<binary data>'}")

            # 转换图片数据为字节
            image_bytes = self._convert_to_bytes(image_data)
            
            # 创建识别链
            recognition_chain = self._create_recognition_chain()
            
            # 执行识别
            logger.info("正在进行图片识别...")
            recognition_response = recognition_chain.invoke({"image_data_list": [image_bytes]})
            recognition_result = recognition_response.content if hasattr(recognition_response, 'content') else recognition_response
            
            if self.debug:
                logger.info("图片识别完成，正在进行文字处理...")
                
            # 创建分析链
            analysis_chain = self._create_analysis_chain()
            
            # 执行分析
            analysis_response = analysis_chain.invoke({"recognition_result": recognition_result})
            analysis_result = analysis_response.content if hasattr(analysis_response, 'content') else analysis_response
            
            if self.debug:
                logger.info("分析完成")
                
            # 返回结果
            return {
                "status": "success",
                "image_url": image_data if isinstance(image_data, str) else None,
                "recognition_result": recognition_result,
                "analysis_result": analysis_result
            }
            
        except Exception as e:
            error_msg = f"处理图片时出现错误: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "image_url": image_data if isinstance(image_data, str) else None,
                "error": str(e)
            }

    def process_multiple_images(self, image_data_list: List[Union[str, bytes, io.BytesIO, Image.Image]], stream: bool = False) -> Dict[str, Any]:
        """
        批量处理多张图片
        
        Args:
            image_data_list: 图片数据列表，每个元素可以是URL、二进制数据、BytesIO对象或PIL.Image对象
            stream: 是否使用流式输出
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            if self.debug:
                logger.info(f"开始批量处理 {len(image_data_list)} 张图片")

            # 处理所有图片数据
            processed_images = []
            for i, image_data in enumerate(image_data_list, 1):
                if self.debug:
                    logger.info(f"正在准备第 {i}/{len(image_data_list)} 张图片")
                
                # 转换图片数据为字节
                image_bytes = self._convert_to_bytes(image_data)
                processed_images.append(image_bytes)

            if not processed_images:
                raise ValueError("没有成功处理任何图片")

            # 创建识别链
            recognition_chain = self._create_recognition_chain()
            
            # 执行识别
            logger.info("正在进行图片识别...")
            recognition_response = recognition_chain.invoke({"image_data_list": processed_images})
            recognition_result = recognition_response.content if hasattr(recognition_response, 'content') else recognition_response
            
            # 记录识别结果到日志文件
            logger.debug("=== 图像识别输出结果 ===")
            logger.debug(recognition_result)
            
            if self.debug:
                logger.info("图片识别完成，正在进行文字处理...")
                
            # 创建分析链
            analysis_chain = self._create_analysis_chain()
            
            # 执行分析
            analysis_response = analysis_chain.invoke({"recognition_result": recognition_result})
            analysis_result = analysis_response.content if hasattr(analysis_response, 'content') else analysis_response
            
            # 记录分析结果到日志文件
            logger.debug("=== 数据分析输出结果 ===")
            logger.debug(analysis_result)
            
            if self.debug:
                logger.info("批量分析完成")
                
            # 返回结果
            return {
                "status": "success",
                "image_count": len(image_data_list),
                "recognition_result": recognition_result,
                "analysis_result": analysis_result
            }
            
        except Exception as e:
            error_msg = f"批量处理图片时出现错误: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": str(e)
            } 