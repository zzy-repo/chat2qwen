import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from openai import OpenAI
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageAnalyzer:
    """图像分析器类，用于对识别结果进行进一步分析"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                 model_name: str = "qwen-max", debug: bool = True):
        """
        初始化图像分析器
        
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
        
        # 默认分析提示词
        self.default_analysis_prompt = """请对以下图像识别结果进行深入分析。要求如下：
        1. 提取关键信息和数据点，并进行分类整理。
        2. 分析数据之间的关系和趋势。
        3. 提供专业的见解和结论。
        4. 如果发现数据异常或不确定点，请明确指出。
        
        图像识别结果如下：
        {recognition_result}
        """

    def _load_env(self, api_key: Optional[str] = None) -> None:
        """加载环境变量和API密钥"""
        # 如果提供了api_key，直接使用
        if api_key:
            self.api_key = api_key
            return
            
        # 否则从环境变量获取
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("API密钥未设置，请提供api_key参数或设置DASHSCOPE_API_KEY环境变量")

    def _setup_client(self) -> None:
        """配置OpenAI客户端"""
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def analyze_result(self, recognition_result: Dict[str, Any], prompt: Optional[str] = None, 
                      stream: bool = False) -> Union[str, Dict[str, Any]]:
        """
        分析识别结果
        
        Args:
            recognition_result: 图像识别结果
            prompt: 自定义分析提示词，如果为None则使用默认提示词
            stream: 是否使用流式输出
            
        Returns:
            Union[str, Dict[str, Any]]: 分析结果，如果stream=True则返回字符串，否则返回字典
        """
        try:
            if recognition_result.get("status") == "error":
                logger.error(f"识别结果包含错误: {recognition_result.get('error')}")
                return {
                    "status": "error",
                    "error": f"无法分析包含错误的识别结果: {recognition_result.get('error')}"
                }
            
            # 获取识别内容
            recognition_content = recognition_result.get("content", "")
            
            # 准备分析提示词
            if prompt:
                # 如果提供了自定义提示词，替换其中的占位符
                analysis_prompt = prompt.format(recognition_result=recognition_content)
            else:
                # 使用默认提示词，替换其中的占位符
                analysis_prompt = self.default_analysis_prompt.format(recognition_result=recognition_content)
            
            if self.debug:
                logger.info("准备发送分析请求...")
            
            # 构建请求
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': '你是一个专业的数据分析助手，擅长对图像识别结果进行深入分析。'},
                    {'role': 'user', 'content': analysis_prompt}
                ],
                stream=stream
            )
            
            if self.debug:
                logger.info("分析请求成功")
            
            # 处理响应
            if stream:
                full_response = ""
                print("\n正在接收分析结果...")
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
                    logger.info(f"分析结果: {json.dumps(result, ensure_ascii=False)}")
                
                # 返回结构化的结果
                return {
                    "status": "success",
                    "recognition_result": recognition_result,
                    "analysis": result,
                    "model": self.model_name
                }
            
        except Exception as e:
            error_msg = f"分析过程中出现错误: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": str(e)
            }
    
    def save_analysis_result(self, analysis_result: Dict[str, Any], output_path: str, 
                           format: str = "json") -> str:
        """
        保存分析结果到文件
        
        Args:
            analysis_result: 分析结果
            output_path: 输出文件路径
            format: 输出格式，支持"json"或"txt"
            
        Returns:
            str: 保存的文件路径
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if format.lower() == "json":
                # 保存为JSON格式
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(analysis_result, f, ensure_ascii=False, indent=2)
            else:
                # 保存为文本格式
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(f"图像路径: {analysis_result.get('recognition_result', {}).get('image_path', '未知')}\n")
                    f.write(f"识别模型: {analysis_result.get('recognition_result', {}).get('model', '未知')}\n")
                    f.write(f"分析模型: {analysis_result.get('model', '未知')}\n")
                    f.write("\n=== 识别结果 ===\n")
                    f.write(analysis_result.get('recognition_result', {}).get('content', '无内容'))
                    f.write("\n\n=== 分析结果 ===\n")
                    f.write(analysis_result.get('analysis', '无分析结果'))
            
            logger.info(f"分析结果已保存到: {output_path}")
            return output_path
            
        except Exception as e:
            error_msg = f"保存分析结果失败: {str(e)}"
            logger.error(error_msg)
            raise 