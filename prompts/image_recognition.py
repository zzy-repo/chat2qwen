from langchain_core.prompts import ChatPromptTemplate

# 图像识别提示词模板
IMAGE_RECOGNITION_TEMPLATE = """请仔细观察并描述图片内容。要求如下：

1. 详细解释图片中所有数据或表格所表达的含义，确保内容准确、逻辑清晰。
2. 以分层结构输出表格内容。每一层代表一个单元格合并区域或单元项，采用缩进或分组表示层级关系。
3. 附加必要的行合并/列合并说明，但保持语义连贯。
4. 禁止碎片化输出，不可遗漏任何单元格信息。"""

image_recognition_prompt = ChatPromptTemplate.from_template(IMAGE_RECOGNITION_TEMPLATE) 