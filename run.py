#!/usr/bin/env python
import os
import json
import glob
from typing import Dict, Any, Optional, List, Union, Tuple
from pdf2image import convert_from_path
from PIL import Image
import time
from dotenv import load_dotenv
import io

from src.models.config import ModelConfig
from src.chains.image_processing import ImageProcessingChain
from src.utils.logger import logger

def get_api_key() -> str:
    """从环境变量获取API密钥"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
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

def convert_pdf_to_images(pdf_path: str) -> List[Tuple[bytes, int]]:
    """将PDF文件转换为图片字节数据"""
    try:
        images = convert_from_path(pdf_path)
        converted_images = []
        for i, img in enumerate(images):
            # 将图片转换为字节数据
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            converted_images.append((img_byte_arr.getvalue(), i+1))
        return converted_images
    except Exception as e:
        logger.error(f"PDF转换失败: {str(e)}")
        raise

def get_input_files() -> List[Tuple[bytes, Optional[int]]]:
    """获取input文件夹中的所有图片和PDF文件"""
    input_dir = "input"
    os.makedirs(input_dir, exist_ok=True)
    
    result_files = []
    
    # 获取所有图片文件
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']:
        for file_path in glob.glob(os.path.join(input_dir, ext)) + glob.glob(os.path.join(input_dir, ext.upper())):
            try:
                with open(file_path, 'rb') as f:
                    image_bytes = f.read()
                result_files.append((image_bytes, None))
            except Exception as e:
                logger.error(f"读取图片文件 {file_path} 失败: {str(e)}")
                continue
    
    # 获取所有PDF文件
    for pdf_path in glob.glob(os.path.join(input_dir, "*.pdf")) + glob.glob(os.path.join(input_dir, "*.PDF")):
        try:
            converted_images = convert_pdf_to_images(pdf_path)
            result_files.extend(converted_images)
        except Exception as e:
            logger.error(f"处理PDF文件 {pdf_path} 失败: {str(e)}")
            continue
    
    return sorted(result_files, key=lambda x: x[1] if x[1] is not None else 0)

def save_result(result: Dict[str, Any], output_dir: str = "output", output_format: str = "json") -> None:
    """保存处理结果"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        base_name = f"result_{int(time.time())}"
        output_path = os.path.join(output_dir, f"{base_name}.{output_format}")
        
        if output_format.lower() == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n=== 识别结果 ===\n")
                f.write(result.get('recognition_result', '无内容'))
                f.write("\n\n=== 分析结果 ===\n")
                f.write(result.get('analysis_result', '无分析结果'))
                
        logger.info(f"结果已保存到: {output_path}")
        
    except Exception as e:
        logger.error(f"保存结果失败: {str(e)}")
        raise

def main():
    """主函数"""
    try:
        # 获取API密钥
        api_key = get_api_key()
        
        # 初始化配置
        config = ModelConfig(api_key=api_key)
        
        # 初始化图像处理器
        processor = ImageProcessingChain(config)
        
        # 获取输入文件
        input_files = get_input_files()
        if not input_files:
            logger.warning("在input文件夹中未找到任何文件")
            return
            
        logger.info(f"在input文件夹中找到 {len(input_files)} 个文件")
        
        # 收集所有图片数据
        image_data_list = []
        for image_data, page_number in input_files:
            try:
                if page_number is not None:
                    logger.info(f"正在读取PDF第 {page_number} 页")
                else:
                    logger.info("正在读取图片文件")
                image_data_list.append(image_data)
            except Exception as e:
                logger.error(f"读取文件时出错: {str(e)}")
                continue
        
        if image_data_list:
            # 批量处理所有图片
            logger.info(f"已成功读取 {len(image_data_list)} 个文件，正在统一打包给大模型进行分析...")
            result = processor.process_multiple_images(image_data_list)
            
            if "error" in result:
                logger.error(result["error"])
                return
            
            # 保存结果
            save_result(result)
        else:
            logger.error("没有成功读取任何文件")
                
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 