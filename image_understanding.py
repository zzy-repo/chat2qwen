import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 调试模式开关
DEBUG = True

class ImageUnderstandingAPI:
    def __init__(self):
        self._load_env()
        self._setup_client()
        
        # 静态配置
        self.image_path = os.path.join(os.path.dirname(__file__), "input", "image.jpg")  # 使用绝对路径
        self.prompt = "请提供一下你所解析出来的原始数据"  # 请替换为实际提示文本
        self.model_name = "qwen-vl-max"  # 直接设置模型名称

    def _load_env(self) -> None:
        """加载环境变量"""
        if not load_dotenv():
            raise EnvironmentError("无法加载.env文件")
        
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("环境变量DASHSCOPE_API_KEY未设置")

    def _setup_client(self) -> None:
        """配置OpenAI客户端"""
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def _encode_image(self) -> str:
        """将图片编码为Base64"""
        with open(self.image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def process_image(self) -> str:
        """处理图片并调用API"""
        try:
            # 检查图片是否存在
            if not Path(self.image_path).exists():
                raise FileNotFoundError(f"图片文件不存在: {self.image_path}")
            
            # 编码图片
            base64_image = self._encode_image()
            
            if DEBUG:
                logger.info("准备发送API请求...")
            
            # 构建请求
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                            {'type': 'text', 'text': self.prompt}
                        ]
                    }
                ]
            )
            
            if DEBUG:
                logger.info("API请求成功")
                logger.info(f"响应内容: {response.model_dump_json()}")
            
            # 提取结果
            result = response.choices[0].message.content
            return result
            
        except Exception as e:
            error_msg = f"处理过程中出现错误: {str(e)}"
            logger.error(error_msg)
            raise

def main():
    try:
        input_dir = os.path.join(os.path.dirname(__file__), "input")
        os.makedirs(input_dir, exist_ok=True)
        
        api = ImageUnderstandingAPI()
        result = api.process_image()
        print("\n图像理解结果:")
        print("-" * 50)
        print(result)
        print("-" * 50)
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 