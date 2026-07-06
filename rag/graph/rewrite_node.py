"""
Graph1 — 问题重写节点
=====================
当检索到的文档被评估为"不相关"时，此节点将用户问题改写为更清晰、更易检索的形式。

什么时候会走到这里？
    grade_documents 返回 "rewrite" 时，即检索文档与问题不相关。
    可能原因:
    - 用户问题的表述与文档中的用词不同（"AI" vs "人工智能"）
    - 用户问题的语义不够清晰（"那个东西怎么用"）
    - 原始问题包含太多无关信息干扰了检索

重写后会发生什么？
    rewrite → agent（重新决策）
    agent 拿到新问题后重新判断是否需要检索 → 重新检索 → 重新评估文档
    如果还是不相关，可能会再次 rewrite → 形成搜索循环

    与 Graph2 的区别:
    Graph2 有 transforme_count 计数器，限制最多重写 2 次，
    Graph1 没有显式的循环限制（理论上可以无限循环）。
"""

from langchain_core.messages import HumanMessage

from rag.utils.message_utils import get_last_human_message
from rag.llm_models.embeddings_model import llm
from rag.utils.logger import log


def rewrite(state):
    """
    将用户问题改写为更优的检索查询
    ==============================
    让 LLM 分析问题的语义意图，生成一个更适合向量检索的问题版本。

    参数:
        state (AgentState):
            当前图状态，包含完整的 messages 列表。
            我们从中提取最后一条人类消息作为需要重写的原始问题。

    返回:
        dict: {"messages": [HumanMessage]}
            返回一个新的 HumanMessage，内容是重写后的问题。
            LangGraph 将其追加到 messages 列表后，
            下一个节点 (agent) 会看到这个新问题并重新决策。

    重写策略:
        LLM 会尝试:
        1. 理解问题背后的真实语义意图
        2. 补充隐含的关键词
        3. 改进不准确的表述
        4. 生成更适合向量检索的查询语句

    代码逻辑（4 步）:
        1. 获取最后一条人类消息中的原始问题
        2. 构造一个"请改进问题"的提示
        3. 调用 LLM 生成改进后的问题
        4. 返回包装为 HumanMessage 的新问题
    """
    log.info("---转换查询---")

    # ===== 1. 提取原始问题 =====
    messages = state["messages"]
    question = get_last_human_message(messages).content

    # ===== 2. 构造重写提示 =====
    # 直接构造一个 HumanMessage 列表发给 LLM
    # 注意: 这里没有使用 ChatPromptTemplate，而是用 HumanMessage 直接传递指令
    #
    # 提示词设计思路:
    #   - 要求 LLM "理解潜在的语义意图/含义" → 不只是表面改写，而是语义深化
    #   - 将原始问题放在分隔线之间 → 清晰的视觉边界，避免混淆
    #   - "改进后的问题" → 明确输出目标
    msg = [
        HumanMessage(
            content=(
                f"\n"
                f"分析输入并尝试理解潜在的语义意图/含义。\n"
                f"这是初始问题:\n"
                f"------\n"
                f"{question}\n"
                f"------\n"
                f"请提出一个改进后的问题:"
            )
        )
    ]

    # ===== 3. 调用 LLM 生成改进问题 =====
    # llm.invoke(msg): 直接调用 LLM，不绑定任何工具
    # 因为这里不需要 LLM 做工具调用决策，只需要生成更好的问题文本
    response = llm.invoke(msg)

    # ===== 4. 返回重写后的问题 =====
    # 返回 HumanMessage：让 agent 的 get_last_human_message() 直接拿到新问题，
    # tool_calls 会基于改进后的问题去检索，而不是靠 LLM 从上下文推断
    return {"messages": [HumanMessage(content=response.content)]}
