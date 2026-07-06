"""
Graph2 — 回答质量评估链
=======================
评估生成的回答是否真正解决了用户的问题。

与幻觉检测的关系:
    这是 Graph2 质量保障的第二道关卡:
    1. 第一关 (grade_hallucinations): 回答是否基于事实 → 检查"真实性"
    2. 第二关 (grade_answer): 回答是否解决问题 → 检查"有用性"

    两道关卡的顺序是精心设计的:
    - 先检查是否基于事实，因为如果存在幻觉，回答问题再准确也没意义
    - 再检查是否解决问题，因为即使没有幻觉，回答可能偏题或不完整

评估后的处理（在 graph_2.py 中）:
    - grade == "yes"（有用）: 返回 "useful" → 流程结束 (END)
    - grade == "no"（无用）:  返回 "not useful" → 触发问题优化后重新检索
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag.llm_models.embeddings_model import llm


class GradeAnswer(BaseModel):
    """
    回答质量评估模型
    ================
    评估 LLM 的回答是否解决了用户提出的问题。

    binary_score (str):
        二元评估结果。
        可选值:
            - "yes": 回答确实解决了用户的问题 → 流程成功结束
            - "no":  回答未能解决用户的问题 → 需要优化问题后重新检索

    "yes" 和 "no" 的判断标准:
        - "yes": 回答直接、准确、完整地回应了问题核心
        - "no":  回答偏题、不完整、或虽然没有幻觉但没解答到问题本质
    """
    binary_score: str = Field(description="回答是否解决了问题，'yes'或'no'")


# ===== 创建带结构化输出的 LLM =====
# method="function_calling": 使用 function calling 获取结构化评分
# 为什么不用 json_mode？
#   function_calling 对于简单的二元判断更可靠，
#   模型更容易理解和遵循 function calling 的 schema
structured_llm_grader = llm.with_structured_output(
    GradeAnswer, method="function_calling"
)

# ===== 回答评估提示词 =====
system = """您是一个评估回答是否解决用户问题的评分器
            给出'yes'或'no'的二元评分，'yes'表示回答确实解决了该问题"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        # {question}: 用户的原始问题
        # {generation}: LLM 生成的回答
        ("human", "用户问题:\n\n{question}\n\n生成回答:{generation}"),
    ]
)

# ===== 构建回答评估链 =====
# 使用示例:
#   result = answer_grader_chain.invoke({
#       "question": "什么是反向传播？",
#       "generation": "反向传播是训练神经网络的算法，通过链式法则计算梯度..."
#   })
#   print(result.binary_score)  # → "yes"（回答解决了问题）
answer_grader_chain = answer_prompt | structured_llm_grader
