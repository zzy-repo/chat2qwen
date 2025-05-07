import os
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from dotenv import load_dotenv
import json
import glob
from pdf2image import convert_from_path
import io
from PIL import Image
import traceback
import time

from models.config import ModelConfig
from chains.image_processing import ImageProcessingChain

# 配置日志
class CustomFormatter(logging.Formatter):
    def format(self, record):
        # 如果消息是二进制数据，替换为占位符
        if isinstance(record.msg, bytes):
            record.msg = "<binary data>"
        return super().format(record)

# 配置日志处理器
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def get_api_key() -> str:
    """
    从环境变量获取API密钥
    
    Returns:
        str: API密钥
        
    Raises:
        ValueError: 当API密钥未设置时抛出
    """
    # 首先尝试从环境变量直接获取
    api_key = os.getenv("DASHSCOPE_API_KEY")
    
    # 如果环境变量中没有，再尝试从.env文件加载
    if not api_key and os.path.exists(".env"):
        load_dotenv()
        api_key = os.getenv("DASHSCOPE_API_KEY")
    
    if not api_key:
        raise ValueError(
            "未找到API密钥。请通过以下方式之一设置：\n"
            "1. 设置环境变量 DASHSCOPE_API_KEY\n"
            "2. 在项目根目录创建.env文件并添加 DASHSCOPE_API_KEY=your_key"
        )
    
    return api_key

def parse_analysis_result(analysis_result: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    解析分析结果为JSON格式
    
    Args:
        analysis_result: 分析结果字符串
        
    Returns:
        Union[List[Dict[str, Any]], Dict[str, Any]]: 解析后的JSON数据
    """
    try:
        return json.loads(analysis_result)
    except json.JSONDecodeError:
        logger.warning("无法解析分析结果为JSON格式")
        return {"raw_result": analysis_result}

def print_analysis_result(result: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    """
    打印分析结果
    
    Args:
        result: 分析结果
    """
    print("\n=== 分析结果 ===")
    if isinstance(result, list):
        for i, item in enumerate(result, 1):
            print(f"\n项目 {i}:")
            print(json.dumps(item, ensure_ascii=False, indent=2))
    elif isinstance(result, dict):
        if "raw_result" in result:
            print(result["raw_result"])
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("无法识别的结果格式")

def save_result(result: Dict[str, Any], output_dir: str, output_format: str = "json") -> None:
    """
    保存处理结果
    
    Args:
        result: 处理结果
        output_dir: 输出目录
        output_format: 输出格式，支持"json"或"txt"
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        if "image_url" in result and result["image_url"]:
            image_url = result["image_url"]
            base_name = os.path.splitext(os.path.basename(image_url))[0] or "result"
        else:
            # 如果是PDF转换的图片，使用页码作为文件名的一部分
            if result.get("original_pdf"):
                base_name = f"pdf_page_{result.get('page_number', 'unknown')}"
            else:
                base_name = f"result_{int(time.time())}"
            
        output_path = os.path.join(output_dir, f"{base_name}.{output_format}")
        
        if output_format.lower() == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"图像URL: {result.get('image_url', '未知')}\n")
                if result.get("original_pdf"):
                    f.write(f"PDF页码: {result.get('page_number', '未知')}\n")
                f.write("\n=== 识别结果 ===\n")
                f.write(result.get('recognition_result', '无内容'))
                f.write("\n\n=== 分析结果 ===\n")
                f.write(result.get('analysis_result', '无分析结果'))
                
        logger.info(f"结果已保存到: {output_path}")
        
    except Exception as e:
        logger.error(f"保存结果失败: {str(e)}")
        raise

def get_image_url() -> str:
    """
    获取用户输入的图片URL
    
    Returns:
        str: 图片URL
    """
    while True:
        url = input("请输入图片URL（输入'q'退出）: ").strip()
        if url.lower() == 'q':
            raise SystemExit("用户退出程序")
        if url.startswith(('http://', 'https://')):
            return url
        print("请输入有效的URL（以http://或https://开头）")

def convert_pdf_to_images(pdf_path: str) -> List[Tuple[Image.Image, int]]:
    """
    将PDF文件转换为图片，保持在内存中
    
    Args:
        pdf_path: PDF文件路径
        
    Returns:
        List[Tuple[Image.Image, int]]: 图片对象和页码的元组列表
    """
    try:
        images = convert_from_path(pdf_path)
        return [(img, i+1) for i, img in enumerate(images)]
    except Exception as e:
        logger.error(f"PDF转换失败: {str(e)}")
        raise

def get_input_files() -> List[Tuple[Union[str, Image.Image], Optional[int]]]:
    """
    获取input文件夹中的所有图片和PDF文件
    
    Returns:
        List[Tuple[Union[str, Image.Image], Optional[int]]]: 文件路径或图片对象与页码的元组列表
    """
    input_dir = "input"
    os.makedirs(input_dir, exist_ok=True)
    
    result_files = []
    
    # 获取所有图片文件
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']:
        for file_path in glob.glob(os.path.join(input_dir, ext)) + glob.glob(os.path.join(input_dir, ext.upper())):
            result_files.append((file_path, None))
    
    # 获取所有PDF文件
    for pdf_path in glob.glob(os.path.join(input_dir, "*.pdf")) + glob.glob(os.path.join(input_dir, "*.PDF")):
        try:
            converted_images = convert_pdf_to_images(pdf_path)
            result_files.extend(converted_images)
        except Exception as e:
            logger.error(f"处理PDF文件 {pdf_path} 失败: {str(e)}")
            continue
    
    return sorted(result_files, key=lambda x: x[1] if x[1] is not None else 0)

def process_local_image(processor: ImageProcessingChain, image_data: Union[str, Image.Image], page_number: Optional[int] = None, stream: bool = False) -> Dict[str, Any]:
    """
    处理本地图片文件或内存中的图片对象
    
    Args:
        processor: 图像处理器实例
        image_data: 图片文件路径或图片对象
        page_number: 页码（如果是PDF转换的图片）
        stream: 是否使用流式输出
        
    Returns:
        Dict[str, Any]: 处理结果
    """
    try:
        if isinstance(image_data, str):
            image_url = f"file://{os.path.abspath(image_data)}"
            result = processor.process_image(image_url, stream=stream)
        else:
            img_byte_arr = io.BytesIO()
            image_data.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            result = processor.process_image(img_byte_arr, stream=stream)
        
        if not result or result.get("status") == "error":
            error_msg = result.get("error", "图片处理失败，未获得有效结果")
            raise ValueError(error_msg)
        
        if page_number is not None:
            result['original_pdf'] = True
            result['page_number'] = page_number
            # 为PDF转换的图片添加一个虚拟的image_url
            if not result.get('image_url'):
                result['image_url'] = f"pdf_page_{page_number}.png"
        
        return result
    except Exception as e:
        error_msg = f"处理图片失败: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg,
            "analysis_result": "处理失败",
            "recognition_result": "处理失败"
        }

def main():
    """主函数"""
    try:
        # 获取API密钥
        api_key = get_api_key()
        
        # 初始化配置
        config = ModelConfig(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            recognition_model="qwen-vl-max",
            analysis_model="qwen-max"
        )
        
        # 初始化图像处理链
        processor = ImageProcessingChain(config, debug=True)
        
        # 获取input文件夹中的文件
        input_files = get_input_files()
        
        if input_files:
            logger.info(f"在input文件夹中找到 {len(input_files)} 个文件")
            for image_data, page_number in input_files:
                try:
                    if isinstance(image_data, str):
                        logger.info(f"正在处理文件: {image_data}")
                    else:
                        logger.info(f"正在处理PDF第 {page_number} 页")
                    
                    result = process_local_image(processor, image_data, page_number, stream=False)
                    
                    if "error" in result:
                        logger.error(result["error"])
                        continue
                    
                    analysis_data = parse_analysis_result(result["analysis_result"])
                    print_analysis_result(analysis_data)
                    
                    result["parsed_analysis"] = analysis_data
                    save_result(result, "output", "json")
                except Exception as e:
                    logger.error(f"处理文件时出错: {str(e)}")
                    continue
        else:
            logger.info("input文件夹中没有找到文件，切换到URL输入模式")
            image_url = get_image_url()
            result = process_local_image(processor, image_url, stream=False)
            
            if "error" in result:
                logger.error(result["error"])
                return
            
            analysis_data = parse_analysis_result(result["analysis_result"])
            print_analysis_result(analysis_data)
            
            result["parsed_analysis"] = analysis_data
            save_result(result, "output", "json")
            
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        if config.debug:
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 

