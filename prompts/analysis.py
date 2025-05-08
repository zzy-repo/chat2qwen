from langchain_core.prompts import ChatPromptTemplate

# 分析提示词模板
# 备注：需要对 JSON 示例中的大括号进行转义，使用双大括号 {{ 和 }} 来表示。
ANALYSIS_TEMPLATE = """
请根据识别结果整合数据，仅输出标准 JSON 格式的数据。

字段说明：
1. 单据信息（document_info）：
   - document_title：单据标题
   - document_date：单据日期
   - document_number：单据编号
   - document_type：单据类型
   - document_remarks：单据备注
   - consignor_info：货主信息
   - warehouse_info：仓库信息

2. 货物信息（goods_info）：
   - manufacturer：厂家（包括敬业、承钢、首钢长治、河钢、宁钢等）
   - variety：品种（包括螺纹、盘钢、圆钢、线材等）
   - material：材质
   - specification：规格
   - price：价格
   - supplier：供应商

示例数据如下：
{{
    "document_info": {{
        "document_title": "钢材销售单",
        "document_date": "2024-03-20",
        "document_number": "XS20240320001",
        "document_type": "销售单",
        "document_remarks": "含税价格",
        "consignor_info": "辽宁成大钢铁贸易有限公司",
        "warehouse_info": "沈阳仓库"
    }},
    "goods_info": [{{
        "manufacturer": "北台",
        "variety": "线材",
        "material": "Q235L",
        "specification": "Φ6.5",
        "price": "4200",
        "supplier": "辽宁成大钢铁贸易有限公司"
    }}]
}}

请注意：
- 所有字段统一用英文表示
- 若某字段无内容，请填入 "无"
- 仅输出标准 JSON 格式的数据，禁止输出除 JSON 数据以外的任何内容，包括解释、说明或注释。

图像识别结果如下：
{recognition_result}
"""

analysis_prompt = ChatPromptTemplate.from_template(ANALYSIS_TEMPLATE) 