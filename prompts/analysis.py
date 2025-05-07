from langchain_core.prompts import ChatPromptTemplate

# 分析提示词模板
ANALYSIS_TEMPLATE = """
请根据识别结果整合数据
请注意：
1. 仅输出标准 JSON 格式的数据
2. 不要输出任何解释或备注
3. 不要输出任何其他内容
4. 字段名是英文名

图像识别结果如下：
{recognition_result}
"""

analysis_prompt = ChatPromptTemplate.from_template(ANALYSIS_TEMPLATE) 