"""
Graph1 状态管理模块
===================
定义 Graph1（基础 RAG）工作流的状态数据结构和评分模型。

LangGraph 的状态管理核心概念:
    - 每个节点执行后返回一个字典，LangGraph 自动将该字典更新到全局状态中
    - Annotated 类型配合 reducer 函数，定义了状态字段的"更新策略"
    - add_messages 是一个 reducer，它告诉 LangGraph"追加新消息"而不是"覆盖旧消息"
"""

from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class AgentState(TypedDict):
    """
    Graph1 的全局状态结构
    =====================
    继承自 TypedDict，提供类型安全的字典结构。
    图执行过程中，每个节点的返回值会自动合并到这个状态中。

    字段说明:
        messages (Annotated[list[BaseMessage], add_messages]):
            LangGraph 的消息列表，是图中流转的核心数据。
            - 类型: list[BaseMessage]，即 LangChain 消息对象的列表
            - 更新策略: add_messages（追加模式）

    add_messages 机制详解:
        add_messages 是 LangGraph 内置的消息列表合并函数。
        它不只简单地 append，还有以下智能行为:
        1. 同 ID 消息去重: 如果新消息的 ID 与已有消息相同，会替换而非追加
        2. HumanMessage 追加: 新的人类消息会追加到列表末尾
        3. ToolMessage 匹配: 工具响应消息会匹配到对应的工具调用消息之后
        4. AIMessage 覆盖: 如果 LLM 重新生成（相同 tool_call_id），会替换旧版本

        这意味着:
        - 你不需要手动管理消息列表的长度
        - 多轮对话的消息会自动累积
        - 图状态始终反映最新的对话状态

    BaseMessage 的子类型:
        - HumanMessage: 用户输入的消息
        - AIMessage: AI/LM 生成的消息（可能包含 tool_calls）
        - ToolMessage: 工具执行后的返回结果
        - SystemMessage: 系统提示消息
    """
    messages: Annotated[list[BaseMessage], add_messages]


class Grade(BaseModel):
    """
    文档相关性评分模型
    ==================
    用于 LLM 结构化输出，要求 LLM 必须按此格式返回评分结果。

    为什么用 Pydantic BaseModel 而非普通 dict？
        - 类型安全: 编译时可检查字段类型
        - 自动校验: Pydantic 会验证 LLM 输出是否符合格式
        - 结构化输出: with_structured_output() 能将此模型注入到 LLM 的 function calling 中

    字段说明:
        binary_score (str):
            文档与问题的相关性二元评分。
            Field(description="相关性评分 'yes'或'no'") 有双重作用:
                1. Pydantic 校验: 确保字段存在且类型为 str
                2. LLM 提示: description 会被传递给 LLM，
                   告诉 LLM 这个字段的含义和期望输出

            可选值:
                - "yes": 文档内容与用户问题相关 → 走 generate 节点生成答案
                - "no":  文档内容与用户问题不相关 → 走 rewrite 节点重写问题后重新检索
    """
    binary_score: str = Field(description="相关性评分 'yes'或'no'")
