from langchain_core.prompts import ChatPromptTemplate

# 分析提示词模板
ANALYSIS_TEMPLATE = """
请根据识别结果整合数据，仅输出标准 JSON 格式的数据，字段顺序为：
- manufacturer（厂家）：包括敬业、承钢、首钢长治、河钢、宁钢等
- variety（品种）：包括螺纹、盘钢、圆钢、线材等
- material（材质）
- specification（规格）
- price（价格）
- supplier（供应商）

请注意：
- 所有字段统一用英文表示
- 若某字段无内容，请填入 "无"
- 仅输出标准 JSON 格式的数据，禁止输出除 JSON 数据以外的任何内容，包括解释、说明或注释。

图像识别结果如下：
{recognition_result}
"""

analysis_prompt = ChatPromptTemplate.from_template(ANALYSIS_TEMPLATE) 