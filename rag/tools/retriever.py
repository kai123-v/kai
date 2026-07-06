"""
检索工具模块
============
创建基于 Milvus 向量数据库的检索工具，供 Agent 和 Graph 节点使用。

检索器参数:
    search_type="similarity": 余弦相似度检索
    k=3:                    返回文档数量上限
    score_threshold=0.1:    相似度最低阈值
    ranker_type="rrf":      混合检索融合算法 (Reciprocal Rank Fusion)
    filter={"category": "content"}: 只检索正文内容，过滤纯标题
"""

from rag.documents.milvus_db import MilvusVectorSave
from langchain_core.tools import tool


# ==================== 初始化 Milvus 连接 ====================
_mv = MilvusVectorSave()
_mv.create_connection()

_retriever = _mv.vector_store_saved.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.1,
        "ranker_type": "rrf",
        "ranker_params": {"k": 100},
        "filter": {"category": "content"},
    },
)


# ==================== 检索工具 ====================

@tool
def retriever_tool(query: str) -> str:
    """搜索关于人工智能、机器学习、深度学习、数学基础等相关知识"""
    return _retriever.invoke(query)


# ==================== 对外导出 ====================
# retriever_tool: 同时供两处使用：
#   - Graph1: llm.bind_tools([retriever_tool]) + ToolNode([retriever_tool])
#   - Graph2: retriever_node 中 retriever_tool.invoke(query) 直接检索
