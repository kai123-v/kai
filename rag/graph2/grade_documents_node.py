"""
Graph2 — 文档过滤节点
=====================
遍历检索到的文档列表，对每个文档进行相关性评分，仅保留相关文档。

这是 Graph2 相比 Graph1 的一个重要改进:
    Graph1: 将整个检索结果作为整体评估 → 全有或全无
    Graph2: 逐文档评估 → 可以部分保留、部分丢弃

好处:
    假设检索返回 3 个文档:
    - 文档1: 与问题高度相关 → 保留
    - 文档2: 部分相关 → 保留（宽松策略）
    - 文档3: 完全不相关 → 丢弃
    最终用 2 个相关文档生成答案，比全丢弃好得多。
"""

from rag.graph2.grade_chain import retrieval_grade_chain
from rag.utils.logger import log


def grade_documents(state):
    """
    评估并过滤检索文档
    ==================
    对每个检索到的文档逐一评分，仅保留与问题相关的文档。

    参数:
        state (GraphState):
            当前图状态，关键字段:
            - question (str): 用户问题
            - documents (List[Document]): 待评估的文档列表

    返回:
        dict: 更新后的状态:
            {"documents": filtered_docs, "question": question}
            - documents: 仅包含相关问题文档的列表（过滤后的结果）
            - question: 保持原样传递

    代码逻辑（5 步）:
        1. 从状态获取问题和文档
        2. 初始化 filtered_docs 空列表
        3. 遍历每个文档:
           ├── 调用 retrieval_grade_chain.invoke() 评分
           ├── grade == "yes" → 加入 filtered_docs
           ├── grade != "yes" → 丢弃（不加入列表）
           └── 异常 → 默认保留该文档（避免因评分出错而丢弃可能相关的文档）
        4. 返回过滤后的文档列表

    异常处理策略:
        评分过程中如果 LLM 调用失败（如 API 超时、返回格式异常等），
        默认保留该文档而不是丢弃。这样做的原因:
        - 宁可多保留噪音文档，也不要漏掉相关文档
        - 后续生成节点的 LLM 有一定能力自行过滤无关信息
    """
    log.info("--CHECK DOCUMENT RELEVANCE TO QUESTION")

    # ===== 1. 提取数据 =====
    question = state["question"]
    documents = state["documents"]

    # ===== 2. 逐文档评分过滤 =====
    filtered_docs = []  # 存放通过评分检查的文档

    for doc in documents:
        try:
            # 调用评分链
            # doc.page_content: 文档的文本内容
            # question: 用户问题
            score = retrieval_grade_chain.invoke(
                {"question": question, "document": doc.page_content}
            )
            # 获取二元评分: "yes" 或 "no"
            grade = score.binary_score

        except Exception as e:
            # 评分过程出错（LLM 超时/格式错误等）
            # 策略: 默认保留该文档
            # 原因: 评分函数是"辅助优化"而非"必需过滤"，
            #       出错时保留文档比丢弃更安全
            log.warning(f"--文档评分异常，默认保留该文档: {str(e)[:100]}--")
            filtered_docs.append(doc)
            continue  # 跳过后续判断，继续处理下一个文档

        # ===== 3. 根据评分决定保留或丢弃 =====
        if grade == "yes":
            # 条件成立: 文档与问题相关 → 保留
            log.info("--GRADE:相关--")
            filtered_docs.append(doc)
        else:
            # 条件不成立: 文档与问题不相关 → 丢弃（不加入 filtered_docs）
            log.info("--GRADE:不相关--")

    # ===== 4. 返回过滤结果 =====
    # filtered_docs: 可能为 0 个（全不相关）、1-3 个（部分相关）、或全部
    # 下游 decide_to_generate 函数会根据 filtered_docs 的数量决定下一步
    return {"documents": filtered_docs, "question": question}
