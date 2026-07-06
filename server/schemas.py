"""
请求/响应数据模型（Pydantic Schemas）
=====================================
使用 Pydantic 定义 API 请求和响应的数据结构，自动完成：
fast api写法。对象继承自basemodel。fast api通过读取装饰器解析json转换为python对象，（将json数据注入python属性中）
- 请求参数校验（类型、必填/可选）
- API 文档生成（Swagger UI 中展示）
- 请求体解析（自动将 JSON 转换为 Python 对象）
"""

from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    """
    聊天请求模型
    ============
    定义 POST /api/chat 接口的请求体结构。

    属性说明:
        question (str):
            用户输入的问题文本。
            示例: "什么是机器学习？"

        graph_type (str, 默认 "graph2"):
            选择使用的 RAG 工作流管道。
            可选值:
                - "graph1": 基础 RAG 流程
                    Agent 判断是否检索 → 向量检索 → 文档评估 → 生成/重写问题
                    特点: 简单直接，适合知识库内的明确问答
                - "graph2": Corrective RAG 流程（默认）
                    路由决策 → 检索/网络搜索 → 逐文档评估 → 生成答案
                    → 幻觉检测 → 回答质量评估 → 自动修正循环
                    特点: 多轮修正，有幻觉检测，适合复杂或需要联网的问题
            为什么默认 graph2？
                graph2 功能更完善，有幻觉检测和自动纠错机制，
                对于 graph1 无法处理的场景（如文档不相关）能自动重试。

        thread_id (Optional[str], 默认 None):表示这个字段可以是字符串或者None
            会话唯一标识，用于多轮对话记忆。
            可选值:
                - None: 不传入或传 null，后端会自动生成新的 UUID 作为会话 ID，
                        此时开启一个全新的对话。
                        后续如果想继续这个对话，需要保存返回的 thread_id。
                - 已有的 UUID 字符串: 传入之前对话的 thread_id，
                        后端会加载该会话的历史消息，实现多轮对话。
            使用场景:
                - 首次对话: 不传 thread_id，从返回的 session 事件中获取新 ID
                - 继续对话: 传入上次的 thread_id，AI 会记得之前的上下文
    """

    question: str
    graph_type: str = "graph2"  # 默认使用 Corrective RAG（功能更完善）
    thread_id: Optional[str] = None  # None 表示新会话，传入 ID 表示继续已有会话
