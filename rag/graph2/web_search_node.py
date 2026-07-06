"""
Graph2 — 网络搜索节点
=====================
使用百度 AI 搜索 API 进行联网检索，获取向量库之外的信息。

何时会被调用？
    1. route_question 判断问题不在向量库范围内 → 直接走网络搜索
    2. grade_documents 后无相关文档，且 transforme_count >= 2 → 降级为网络搜索

与 retriever_node 的并行关系:
    web_search 和 retrieve 是两条并行的检索路径：
    - web_search: 搜索互联网 → 信息最新、最广，但速度慢、内容不可控
    - retrieve: 搜索向量库 → 速度快、内容可控，但知识范围有限
    两者在 generate 节点汇合，使用相同的生成逻辑处理搜索结果。
"""

from langchain_core.documents import Document

from rag.tools.baidu_search_tool import baidu_search_tool
from rag.utils.logger import log


def web_search(state):
    """
    基于用户问题进行网络搜索
    ========================
    调用百度 AI 搜索 API 获取互联网上的相关信息。

    参数:
        state (GraphState):
            当前图状态，关键字段:
            - question (str): 用户问题（可能是原始问题或优化后的问题）

    返回:
        dict: 更新后的状态:
            {"documents": [web_results], "question": question}
            - documents: 包装为 Document 列表的网络搜索结果
            - question: 保持原样传递

    代码逻辑（4 步）:
        1. 提取问题
        2. 调用百度 AI 搜索
        3. 将搜索结果包装为 LangChain Document 对象
        4. 返回更新后的状态

    为什么将结果包装为 Document？
        为了与 retrieve 节点的输出格式统一。
        后续 generate 节点不关心文档来自向量库还是网络搜索，
        它只要求输入是 List[Document] 格式。
        这种"统一接口"是策略模式的应用。

    百度 AI 搜索的特点（在 rag/tools/baidu_search_tool.py 中配置）:
        - search_recency_filter: "month" → 优先返回最近一个月的信息
        - search_source: "baidu_search_v2" → 使用百度搜索 v2 版本
    """
    log.info("--网络搜索--")

    # ===== 1. 提取问题 =====
    question = state["question"]

    # ===== 2. 执行网络搜索 =====
    # baidu_search_tool 是 @tool 装饰器注册的 LangChain Tool
    # invoke({"query": question}): 以问题为查询词进行搜索
    # 返回: 百度 AI 搜索的 JSON 格式结果字符串
    result = baidu_search_tool.invoke({"query": question})

    # ===== 3. 包装为 Document =====
    # 统一的 Document 格式，与向量检索结果保持一致
    # page_content: 搜索结果的原始文本
    # metadata: 可以添加来源等元数据（此处省略，默认为空字典）
    web_results = Document(page_content=str(result))

    # ===== 4. 返回更新状态 =====
    # documents 是单元素列表（网络搜索只返回一个综合结果）
    # 与 retriever_node 返回的格式一致
    return {"documents": [web_results], "question": question}
