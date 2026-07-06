"""
Graph2 — 文档检索节点
=====================
从 Milvus 向量数据库中检索与用户问题最相关的文档。

这是向量检索路径的入口节点，被 route_question 路由到这里。
与 Graph1 的 retrieve 节点的区别:
    Graph1: 使用 ToolNode 自动处理工具调用（因为 Graph1 有 Agent 判断工具调用）
    Graph2: 直接调用 retriever_tool.invoke()，因为 Graph2 没有 Agent 工具调用判断环节，
           而是通过路由直接决定走检索路径。
"""

from rag.tools.retriever import retriever_tool
from rag.utils.logger import log


def retrieve(state):
    """
    搜索相关文档
    ============
    从向量知识库中检索与用户问题语义相似的文档。

    参数:
        state (GraphState):
            当前图状态，关键字段:
            - question (str): 用户问题（可能是原始问题或优化后的问题）

    返回:
        dict: 更新后的状态，包含:
            {"documents": List[Document], "question": str}
            - documents: 检索到的文档列表（最多 3 个，由 retriever 的 k 参数控制）
            - question: 保持原样传回（后续节点可能需要）

    代码逻辑（3 步）:
        1. 从状态中提取问题
        2. 调用 retriever_tool.invoke(question) 进行向量检索
        3. 返回检索结果和问题

    retriever 的检索过程（在 rag/tools/retriever.py 中配置）:
        1. 用 BGE Embedding 将问题转化为 512 维向量
        2. 在 Milvus 中搜索最相似的 k=3 个文档
        3. 使用 RRF 算法融合密集向量和稀疏向量(BM25)的排序结果
        4. 只返回相似度 > 0.1 的文档
        5. 只返回 category="content" 的文档（过滤掉纯标题等非内容文档）
    """
    log.info("---检索库检索---")

    # ===== 1. 提取问题 =====
    question = state["question"]

    # ===== 2. 执行检索 =====
    # retriever 是 Milvus 向量库的检索器，配置见 rag/tools/retriever.py
    # invoke(question): 以问题文本作为查询，返回最匹配的 Document 列表
    documents = retriever_tool.invoke(question)

    # ===== 3. 返回更新状态 =====
    # documents: 检索到的文档列表
    # question: 保持原样传递，后续节点（grade_documents）需要知道原始问题
    return {"documents": documents, "question": question}
