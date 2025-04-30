import os
import logging
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv
import json

from models.config import ModelConfig
from chains.image_processing import ImageProcessingChain

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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
    if not api_key:
        # 检查.env文件是否存在
        if os.path.exists(".env"):
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
        # 尝试直接解析JSON
        parsed = json.loads(analysis_result)
        return parsed
    except json.JSONDecodeError:
        # 如果解析失败，返回原始结果
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
        # 如果是列表，打印每个项目
        for i, item in enumerate(result, 1):
            print(f"\n项目 {i}:")
            print(json.dumps(item, ensure_ascii=False, indent=2))
    elif isinstance(result, dict):
        if "raw_result" in result:
            # 如果是原始结果，直接打印
            print(result["raw_result"])
        else:
            # 如果是字典，格式化打印
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
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        image_url = result["image_url"]
        base_name = os.path.splitext(os.path.basename(image_url))[0] or "result"
        output_path = os.path.join(output_dir, f"{base_name}.{output_format}")
        
        if output_format.lower() == "json":
            # 保存为JSON格式
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            # 保存为文本格式
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"图像URL: {result.get('image_url', '未知')}\n")
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

def main():
    """主函数"""
    try:
        # 获取API密钥
        api_key = get_api_key()
        
        # 设置参数
        output_dir = "output"           # 输出目录
        output_format = "json"          # 输出格式：json 或 txt
        recognition_model = "qwen-vl-max"  # 识别模型名称
        analysis_model = "qwen-max"     # 分析模型名称
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # API基础URL
        debug = True                    # 是否启用调试模式
        stream = False                   # 是否使用流式输出
        
        # 初始化模型配置
        config = ModelConfig(
            api_key=api_key,
            base_url=base_url,
            recognition_model=recognition_model,
            analysis_model=analysis_model
        )
        
        # 初始化图像处理链
        processor = ImageProcessingChain(config, debug=debug)
        
        # 获取图片URL
        image_url = get_image_url()
        
        # 处理图片
        result = processor.process_image(image_url, stream=stream)
        
        # 解析分析结果
        analysis_data = parse_analysis_result(result["analysis_result"])
        
        # 打印分析结果
        print_analysis_result(analysis_data)
        
        # 保存完整结果
        result["parsed_analysis"] = analysis_data
        save_result(result, output_dir, output_format)
            
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 


# http://gme.gusen.steel56.com.cn/gdai/rukudan1.jpg