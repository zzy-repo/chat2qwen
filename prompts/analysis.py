from langchain_core.prompts import ChatPromptTemplate

# 分析提示词模板
ANALYSIS_TEMPLATE = """请将报价单内容整理成结构化的json形式，按照⼚家、品种、材质、规格、价格、供应商依次排列。

⼚家包括：敬业、承钢、⾸钢⻓治、河钢、宁钢，等⼚家
品种包括：螺纹、盘钢、圆钢、线材，等品种

图像识别结果如下：
{recognition_result}"""

analysis_prompt = ChatPromptTemplate.from_template(ANALYSIS_TEMPLATE) 