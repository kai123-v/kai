"""
Graph2 — 文档相关性评分链
=========================
定义用于评估检索文档是否与用户问题相关的 LLM 链。

与 Graph1 的 grade_documents 函数的关系:
    Graph1: 评分逻辑直接写在 graph1.py 中，评估整个检索结果
    Graph2: 评分逻辑抽离为独立的链 (retrieval_grade_chain)，
           然后在 grade_documents_node.py 中遍历每个文档逐一评估
    区别:
    - Graph1: 一次评估整个检索结果 → 粗糙，但快
    - Graph2: 逐文档评估 → 精细，可以过滤部分不相关文档，
              保留相关文档以减少信息丢失
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel

from rag.llm_models.embeddings_model import llm


class GradeDocuments(BaseModel):
    """
    文档相关性评分模型
    ==================
    要求 LLM 对单个文档是否与问题相关给出二元评分。

    binary_score (str):
        二元评分结果。
        Field(description=...) 会作为 LLM function calling 的参数描述。
        可选值:
            - "yes": 文档与问题相关 → 保留此文档
            - "no":  文档与问题不相关 → 丢弃此文档

    评分标准（在系统提示词中说明）:
        - 文档包含关键词或语义含义 → 相关
        - 不需要非常严格的测试 → 允许一定噪音
        - 目的: 过滤掉明显错误的检索结果，而非追求完美精准
    """
    binary_score: str = Field(description="文档是否与问题相关，取值为'yes'或'no'")


# ===== 创建带结构化输出的 LLM =====
# method="function_calling": 使用 OpenAI 兼容的 function calling 机制
# 可选 method 值:
#   - "function_calling": 最常用，LLM 调用一个函数来返回结构化数据
#   - "json_mode": 强制 LLM 输出纯 JSON（OpenAI 支持）
#   - "json_schema": 使用 JSON Schema 约束格式（更严格）
structured_llm_grader = llm.with_structured_output(
    GradeDocuments, method="function_calling"
)

# ===== 评分提示词 =====
system = """你是一个评估检索文档与用户问题相关性的评分器。\n
如果文档包含与用户问题相关的关键词或语义含义，则评为相关。\n
不需要非常严格的测试，目的是过滤掉错误的检索结果\n
给出二元评分'yes' 或'no'来表示文档是否与问题相关。
"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),  # 系统角色：设定评分器行为
        # 用户消息: 传入的文档内容和用户问题
        # {document}: 单个文档的 page_content（文本内容）
        # {question}: 用户问题文本
        ("human", "Retrieved document:\n\n{document}\n\nUser {question}"),
    ]
)

# ===== 构建评分链 =====
# grade_prompt: 将输入格式化为提示词
# structured_llm_grader: LLM 处理并返回 GradeDocuments 对象
# 使用示例:
#   result = retrieval_grade_chain.invoke({
#       "document": "机器学习是人工智能的一个分支...",
#       "question": "什么是机器学习？"
#   })
#   print(result.binary_score)  # → "yes"
retrieval_grade_chain = grade_prompt | structured_llm_grader
