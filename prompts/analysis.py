from langchain_core.prompts import ChatPromptTemplate

# 分析提示词模板
ANALYSIS_TEMPLATE = """
请根据以下图像识别结果，仅输出标准 JSON 格式的数据，字段顺序为：厂家、品种、材质、规格、价格、供应商。
•	厂家包括：敬业、承钢、首钢长治、河钢、宁钢等
•	品种包括：螺纹、盘钢、圆钢、线材等
•	若某字段无内容，请填入 "无"
禁止输出除 JSON 数据以外的任何内容，包括解释、说明或注释。

图像识别结果如下：
{recognition_result}
"""

analysis_prompt = ChatPromptTemplate.from_template(ANALYSIS_TEMPLATE) 