"""
Graph2 — 幻觉检测链
===================
检测 LLM 生成的回答是否基于检索到的文档（事实集），
还是"凭空编造"的内容（幻觉）。

什么是"幻觉"（Hallucination）？
    在大语言模型领域，幻觉指模型生成了听起来合理但并非基于事实的内容。
    在 RAG 场景中，幻觉表现为:
    - 回答内容与检索到的文档不一致
    - 回答包含了文档中没有的信息
    - 回答"编造"了看似相关但实际不存在的细节

为什么需要幻觉检测？
    即使给了 LLM 检索文档，它仍可能在以下情况下产生幻觉:
    - 文档信息不足以回答问题时，LLM 倾向于"补充"缺失信息
    - LLM 的预训练知识与文档内容冲突时
    - LLM 过度"联想"，添加了文档中不存在的内容

检测后的处理（在 graph_2.py 中）:
    - grade == "yes"（无幻觉）: 进入回答质量评估
    - grade == "no"（有幻觉）: 返回 "not supported"，触发重新生成
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel
from rag.llm_models.embeddings_model import llm


class GradeHallucinations(BaseModel):
    """
    幻觉检测结果模型
    ================
    要求 LLM 判断生成内容是否基于检索文档（事实集）。

    binary_score (str):
        二元判定结果。
        可选值:
            - "yes": 回答是基于/支持于给定事实集的（无幻觉）
            - "no":  回答中存在事实集不支持的内容（有幻觉）

    注意: 这里 "yes" 表示"无幻觉"（基于事实），"no" 表示"有幻觉"。
    这个命名可能初始让人迷惑，但它和 grade_answer_chain 保持了一致的约定:
    "yes" = 好，"no" = 不好。
    """
    binary_score: str = Field(description="回答是否基于事实，取值为'yes'或'no'")


# ===== 创建带结构化输出的 LLM =====
# method="function_calling": 使用 function calling 机制获取结构化输出
structured_llm_grade = llm.with_structured_output(
    GradeHallucinations, method="function_calling"
)

# ===== 幻觉检测提示词 =====
system = """您是一个评估生成内容是否基于检索事实的评分器。\n
给出'yes'或'no'的二元评分，
'yes'表示回答是基于/支持于给定事实集的。"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        # {documents}: 检索到的文档内容（"事实集"）
        # {generation}: LLM 生成的回答内容
        ("human", "事实集:\n\n{documents}\n\n 生成内容:{generation}"),
    ]
)

# ===== 构建幻觉检测链 =====
# 使用示例:
#   result = hallucination_grader_chain.invoke({
#       "documents": "反向传播是训练神经网络的核心算法...",
#       "generation": "反向传播通过计算损失函数的梯度来更新权重"
#   })
#   print(result.binary_score)  # → "yes"（回答与文档一致）
hallucination_grader_chain = hallucination_prompt | structured_llm_grade
