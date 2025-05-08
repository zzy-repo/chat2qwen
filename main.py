import os
from typing import Dict, Any, Optional, List, Union, Tuple
from dotenv import load_dotenv
import json
import glob
from pdf2image import convert_from_path
import io
from PIL import Image
import traceback
import time
import argparse
import sys
import base64

from models.config import ModelConfig
from chains.image_processing import ImageProcessingChain
from utils.logger import logger

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
            # 直接读取本地文件
            with open(image_data, 'rb') as f:
                image_bytes = f.read()
            result = processor.process_image(image_bytes, stream=stream)
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

def process_image(processor: ImageProcessingChain, image_data: Union[str, bytes], stream: bool = False) -> Dict[str, Any]:
    """
    处理图片数据
    
    Args:
        processor: 图像处理器实例
        image_data: 图片URL或图片二进制数据
        stream: 是否使用流式输出
        
    Returns:
        Dict[str, Any]: 处理结果
    """
    try:
        if isinstance(image_data, str):
            # 处理URL
            if image_data.startswith(('http://', 'https://')):
                result = processor.process_image(image_data, stream=stream)
            else:
                # 处理base64编码的图片数据
                try:
                    image_bytes = base64.b64decode(image_data)
                    result = processor.process_image(image_bytes, stream=stream)
                except Exception as e:
                    raise ValueError(f"无效的图片数据格式: {str(e)}")
        else:
            # 处理二进制数据
            result = processor.process_image(image_data, stream=stream)
        
        if not result or result.get("status") == "error":
            error_msg = result.get("error", "图片处理失败，未获得有效结果")
            raise ValueError(error_msg)
        
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
        # 解析命令行参数
        parser = argparse.ArgumentParser(
            description='图片分析工具',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例：
  # 处理在线图片
  python main.py --url "https://example.com/image.jpg"
  
  # 处理Base64编码的图片
  python main.py --image "base64_encoded_image_data"
  
  # 使用流式输出
  python main.py --url "https://example.com/image.jpg" --stream
  
  # 从标准输入读取
  echo "https://example.com/image.jpg" | python main.py
  """
        )
        parser.add_argument('--url', type=str, help='指定图片URL（http/https）')
        parser.add_argument('--image', type=str, help='指定Base64编码的图片数据')
        parser.add_argument('--stream', action='store_true', help='启用流式输出模式')
        args = parser.parse_args()

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
        
        # 处理输入
        if args.url:
            result = process_image(processor, args.url, stream=args.stream)
        elif args.image:
            result = process_image(processor, args.image, stream=args.stream)
        else:
            # 从标准输入读取图片数据
            print("请输入图片URL或Base64编码的图片数据（输入完成后按Ctrl+D结束）：")
            image_data = sys.stdin.read().strip()
            if image_data.startswith(('http://', 'https://')):
                result = process_image(processor, image_data, stream=args.stream)
            else:
                result = process_image(processor, image_data, stream=args.stream)
        
        if "error" in result:
            logger.error(result["error"])
            return
        
        # 解析并打印分析结果
        analysis_data = parse_analysis_result(result["analysis_result"])
        print_analysis_result(analysis_data)
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        if config.debug:
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 

