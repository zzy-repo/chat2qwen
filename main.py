import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import json

from recognizer.image_recognizer import ImageRecognizer
from analyzer.image_analyzer import ImageAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """图像处理器类，用于协调识别和分析流程"""
    
    def __init__(self, api_key: Optional[str] = None, 
                 recognition_model: str = "qwen-vl-max",
                 analysis_model: str = "qwen-max",
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 debug: bool = True):
        """
        初始化图像处理器
        
        Args:
            api_key: API密钥，如果为None则从环境变量加载
            recognition_model: 识别模型名称
            analysis_model: 分析模型名称
            base_url: API基础URL
            debug: 是否启用调试模式
        """
        self.debug = debug
        
        # 初始化识别器和分析器
        self.recognizer = ImageRecognizer(
            api_key=api_key,
            base_url=base_url,
            model_name=recognition_model,
            debug=debug
        )
        
        self.analyzer = ImageAnalyzer(
            api_key=api_key,
            base_url=base_url,
            model_name=analysis_model,
            debug=debug
        )
    
    def process_image(self, image_path: str, output_dir: str, 
                     output_format: str = "json", 
                     recognition_prompt: Optional[str] = None,
                     analysis_prompt: Optional[str] = None,
                     stream: bool = True) -> Dict[str, Any]:
        """
        处理单张图片
        
        Args:
            image_path: 图片文件路径
            output_dir: 输出目录
            output_format: 输出格式，支持"json"或"txt"
            recognition_prompt: 自定义识别提示词
            analysis_prompt: 自定义分析提示词
            stream: 是否使用流式输出
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            logger.info(f"开始处理图片: {image_path}")
            
            # 识别图片
            recognition_result = self.recognizer.recognize_image(
                image_path, 
                prompt=recognition_prompt,
                stream=stream
            )
            
            if isinstance(recognition_result, str):
                # 如果是流式输出，直接返回识别结果
                return {"status": "success", "content": recognition_result}
            
            if recognition_result.get("status") == "error":
                logger.error(f"图片识别失败: {recognition_result.get('error')}")
                return recognition_result
            
            # 分析识别结果
            analysis_result = self.analyzer.analyze_result(
                recognition_result,
                prompt=analysis_prompt,
                stream=stream
            )
            
            if isinstance(analysis_result, str):
                # 如果是流式输出，直接返回分析结果
                return {"status": "success", "content": analysis_result}
            
            if analysis_result.get("status") == "error":
                logger.error(f"结果分析失败: {analysis_result.get('error')}")
                return analysis_result
            
            # 打印分析结果
            self._print_analysis_result(analysis_result, output_format)
            
            logger.info(f"图片处理完成: {image_path}")
            return analysis_result
            
        except Exception as e:
            error_msg = f"处理图片时出现错误: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "image_path": image_path,
                "error": str(e)
            }
    
    def _print_analysis_result(self, analysis_result: Dict[str, Any], output_format: str) -> None:
        """
        在命令行打印分析结果
        
        Args:
            analysis_result: 分析结果
            output_format: 输出格式，支持"json"或"txt"
        """
        print("\n" + "="*50)
        print("分析结果:")
        print("="*50)
        
        if output_format.lower() == "json":
            # 以JSON格式打印
            print(json.dumps(analysis_result, ensure_ascii=False, indent=2))
        else:
            # 以文本格式打印
            print(f"图像路径: {analysis_result.get('recognition_result', {}).get('image_path', '未知')}")
            print(f"识别模型: {analysis_result.get('recognition_result', {}).get('model', '未知')}")
            print(f"分析模型: {analysis_result.get('model', '未知')}")
            print("\n=== 识别结果 ===")
            print(analysis_result.get('recognition_result', {}).get('content', '无内容'))
            print("\n=== 分析结果 ===")
            print(analysis_result.get('analysis', '无分析结果'))
        
        print("="*50 + "\n")

def main():
    """主函数"""
    # 加载环境变量
    load_dotenv()
    
    # 从环境变量获取API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("未找到API密钥，请确保.env文件中包含DASHSCOPE_API_KEY")
        return
    
    # 设置参数
    input_path = "input/image.png"  # 输入图片路径
    output_dir = "output"           # 输出目录
    output_format = "json"          # 输出格式：json 或 txt
    recognition_model = "qwen-vl-max"  # 识别模型名称
    analysis_model = "qwen-max"     # 分析模型名称
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # API基础URL
    debug = True                    # 是否启用调试模式
    stream = False                   # 是否使用流式输出
    
    # 自定义提示词（可选）
    recognition_prompt = """
    请以分层结构输出表格内容。每一层代表一个单元格合并区域或单元项，采用缩进或分组表示层级关系。附加必要的行合并/列合并说明，但保持语义连贯。禁止碎片化输出，不可遗漏任何单元格信息。
    """
    
    # 注意：这里的 {recognition_result} 是一个占位符，会在运行时被替换
    analysis_prompt = """
    请将报价单内容整理成结构化的表格形式，按照⼚家、品种、材质、规格、价格、供应商依次排列，
    ⼚家包括：敬业、承钢、⾸钢⻓治、河钢、宁钢，等⼚家
    品种包括：螺纹、盘钢、圆钢、线材，等品种
    
    图像识别结果如下：
    {recognition_result}
    """
    
    try:
        # 初始化图像处理器
        processor = ImageProcessor(
            api_key=api_key,
            recognition_model=recognition_model,
            analysis_model=analysis_model,
            base_url=base_url,
            debug=debug
        )
        
        # 检查输入路径是否存在
        if not os.path.exists(input_path):
            logger.error(f"输入路径不存在: {input_path}")
            return
            
        # 处理图片
        processor.process_image(
            input_path, 
            output_dir, 
            output_format,
            recognition_prompt=recognition_prompt,
            analysis_prompt=analysis_prompt,
            stream=stream
        )
            
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 