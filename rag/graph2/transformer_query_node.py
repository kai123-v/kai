"""
Graph2 — 问题优化节点
=====================
将用户问题重写为更适合向量数据库检索的优化版本。

与 Graph1 rewrite_node 的关键区别:
    1. 状态更新方式:
       Graph1: 返回新的 HumanMessage 放入 messages 列表
       Graph2: 直接更新 state 中的 question 字段

    2. 循环控制:
       Graph1: 没有计数器（理论上可能无限循环 rewrite → agent → retrieve → grade → rewrite）
       Graph2: 维护 transforme_count 计数器，最多优化 2 次

    3. 触发条件:
       Graph1: grade_documents 返回 "rewrite" → 总是触发
       Graph2: decide_to_generate 判断无相关文档 且 transforme_count < 2 → 触发
               如果 transforme_count >= 2 且仍无相关文档 → 走 web_search

为什么限制 2 次？
    防止无限循环。大多数情况下，1-2 次问题重写足以改善检索效果。
    如果 2 次后仍然找不到相关文档，说明问题可能不在向量库的知识范围内，
    此时改为网络搜索获取信息更合理。
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.llm_models.embeddings_model import llm
from rag.utils.logger import log


def transformer_query(state):
    """
    优化用户问题，生成更适合检索的查询语句
    ======================================
    让 LLM 理解问题的深层语义意图，生成一个能更好地匹配文档库的查询。

    参数:
        state (GraphState):
            当前图状态，关键字段:
            - question (str): 需要优化的原始问题
            - documents (List[Document]): 当前文档列表（保持原样传回）
            - transforme_count (int): 已优化次数

    返回:
        dict: 更新后的状态:
            {
                "documents": documents,         # 保持原样
                "question": better_question,    # 优化后的问题
                "transforme_count": count + 1,  # 计数 +1
            }

    优化策略:
        LLM 会将问题转换为更适合向量数据库检索的形式:
        - 补充相关的专业术语（如"那个算法" → "反向传播算法"）
        - 修正可能引起歧义的表述
        - 简化为核心语义特征

    异常处理:
        如果 LLM 调用失败，返回原始问题，避免流程中断。
        但 transforme_count 仍然 +1（因为这是一次"尝试"）。
    """
    log.info("--优化问题--")

    # ===== 1. 提取当前状态 =====
    question = state["question"]  # 当前问题（可能是原始问题或已优化过的问题）
    documents = state["documents"]  # 文档列表（不修改，原样传回）
    transforme_count = state.get("transforme_count", 0)  # 获取优化计数器，默认为 0

    # ===== 2. 构建优化提示词 =====
    system = """作为问题重写器，您需要将输入问题转换为更适合向量数据库检索的优化版本
                 请分析输入问题并理解其背后的语义意图/真实含义"""

    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),  # 系统角色：问题优化器
            (
                "human",
                "这是初始问题:\n\n{question}请生成一个优化后的问题",
            ),
        ]
    )

    # ===== 3. 构建优化处理链 =====
    # StrOutputParser(): 将 LLM 的 AIMessage 输出转换为纯字符串
    # 因为我们需要的是问题的文本，而不是消息对象
    question_rewriter = (
        re_write_prompt   # 格式化提示词
        | llm             # LLM 生成优化后的问题
        | StrOutputParser()  # 提取纯文本
    )

    # ===== 4. 执行问题优化 =====
    try:
        better_question = question_rewriter.invoke({"question": question})
    except Exception as e:
        # 优化失败时的降级策略: 保留原问题
        # 为什么不抛异常？因为问题优化是优化步骤，不是必需步骤。
        # 用原问题继续处理，虽然检索效果可能差一些，
        # 但至少流程不中断，用户最终能得到回答。
        log.warning(f"--问题重写异常，保留原问题: {str(e)[:100]}--")
        better_question = question

    # ===== 5. 返回更新后的状态 =====
    # documents: 保持原样
    # question: 替换为优化后的问题
    # transforme_count: 计数器 +1（关键！用于防止无限优化循环）
    #message字段不改。messages消息列表不变，前端只展示ai跟human的消息记录。那么这节点返回的数据没有记录在消息记录。前端不会展示
    return {
        "documents": documents,
        "question": better_question,
        "transforme_count": transforme_count + 1,
    }
