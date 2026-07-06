"""
消息工具函数
============
提供 LangChain 消息列表的通用操作函数，供 Graph1、Graph2 的节点共用。
避免在各个节点之间重复定义相同逻辑，也防止循环导入问题。
"""

from typing import List
from langchain_core.messages import BaseMessage, HumanMessage


def get_last_human_message(messages: List[BaseMessage]) -> HumanMessage:
    """
    获取消息列表中最后一条人类消息
    ==============================
    多轮对话场景中，messages 列表可能包含多轮问答（用户→AI→用户→AI→...）。
    我们需要找到"最新"的用户输入（即列表中最后一条 HumanMessage），
    因为那才是当前要处理的原始问题。

    参数:
        messages (List[BaseMessage]):
            LangChain 消息列表，按对话顺序排列。
            例如: [HumanMessage("什么是AI?"), AIMessage("AI是..."), HumanMessage("详细解释一下")]

    返回:
        HumanMessage: 列表中最后一条用户消息

    异常:
        ValueError: 如果消息列表中没有任何 HumanMessage

    代码逻辑:
        1. 反向遍历消息列表（从最新到最旧）
        2. 检查每个消息是否为 HumanMessage 类型
        3. 第一个匹配的就是最后一条用户消息，直接返回
        4. 如果遍历完都没找到 → 抛出异常（理论上不应该发生，因为至少会有一条用户输入）

    为什么反向遍历而不是取 messages[-1]？
        因为 messages 列表的最后一条不一定是 HumanMessage。
        例如在 agent 节点之后，最后一条可能是 AIMessage 或 ToolMessage。
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    raise ValueError("No Human message found")
