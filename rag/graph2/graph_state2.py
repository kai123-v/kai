"""
Graph2 状态管理模块
===================
定义 Graph2（Corrective RAG）工作流的状态数据结构。

与 Graph1 状态结构的区别:
    Graph1 (AgentState):
        只有一个字段: messages (List[BaseMessage])
        所有信息（问题、检索结果、回答）都放在 messages 列表里
        优点: 简单
        缺点: 节点之间读取特定信息时需要从 messages 中提取，不够直接

    Graph2 (GraphState):
        多个独立字段: question, generation, documents, transforme_count, messages
        每个字段有明确语义，节点可以直接读写特定字段
        优点: 结构清晰，每个节点知道自己读写什么
        缺点: 字段较多，需要在节点之间维护一致性

为什么 Graph2 要独立字段？
    Graph2 流程更复杂，有多个评估和修正步骤:
    - question: 可能被 transformer_query 节点多次重写
    - documents: 可能被 grade_documents 节点过滤
    - generation: 可能被 grade_hallucinations 评估后重新生成
    - transforme_count: 需要跟踪重写次数来防止死循环
    这些如果用 messages 列表管理会很混乱，独立字段更清晰。
"""

from typing import TypedDict, List, Annotated

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class GraphState(TypedDict):
    """
    Graph2 的全局状态结构
    =====================
    表示 Corrective RAG 处理流程中的完整状态信息。

    各字段的作用和生命周期:

        question (str):
            当前处理的用户问题。
            - 初始值: 用户输入的问题
            - 修改节点: transformer_query（问题优化节点）
            - 使用节点: route_question（路由）、retrieve（检索）、
                       web_search（网络搜索）、generate（生成）、
                       grade_generation（评估环节用到原始问题）
            - 说明: 这是一个"可变"字段，会随着问题重写而更新

        transforme_count (int):
            问题优化次数的计数器。
            - 初始值: 0
            - 递增节点: transformer_query（每次优化 +1）
            - 使用节点: decide_to_generate（判断是否需要改为网络搜索）
            - 说明: 防止无限优化循环。最多优化 2 次，
                    超过后如果文档仍不相关，改为网络搜索。

        generation (str):
            LLM 生成的回答文本。
            - 初始值: ""（空字符串）
            - 写入节点: generate（生成节点）
            - 使用节点: grade_generation（幻觉检测 + 回答评估）
            - 说明: 可能被多次重写（幻觉检测不通过时触发重新生成）

        documents (List[Document]):
            检索到的文档列表。
            - 初始值: []（空列表）
            - 写入节点: retrieve（向量检索）、web_search（网络搜索）
            - 修改节点: grade_documents（过滤不相关文档）
            - 使用节点: generate（作为生成回答的参考上下文）
            - 说明: Document 是 LangChain 的文档对象，包含:
                - page_content (str): 文档文本内容
                - metadata (dict): 文档元数据（如来源、标题等）

        messages (Annotated[list[BaseMessage], add_messages]):
            对话历史消息列表。
            - 初始值: [HumanMessage(content=question)]
            - 写入节点: generate（写入 AIMessage 回答）
            - 使用节点: generate（读取对话历史构建上下文）
            - 更新策略: add_messages（追加模式，不是覆盖）
            - 与 Graph1 的区别:
                Graph1 把所有数据都放 messages 中，
                Graph2 中 messages 只用于对话记忆，不作为主要数据传递通道。
    """
    question: str
    transforme_count: int
    generation: str
    documents: List[Document]
    messages: Annotated[list[BaseMessage], add_messages]

