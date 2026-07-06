"""
API 路由模块
===========
restful风格，增用post 查用get，改用put，删用delete
定义所有 REST API 端点，包括：
- POST /api/chat: SSE 流式聊天（核心接口）
- GET  /api/graphs: 获取可用的 RAG 工作流列表
 （fast api+async)异步方式解决高并发请求等待实现流失输出(sse)
"""

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from server.schemas import ChatRequest
from server.stream import stream_graph1, stream_graph2

# 创建路由器实例
# APIRouter 和 FastAPI 类似，但可以独立定义路由然后注册到主应用
# 好处：模块化管理，不同功能的路由可以分文件定义
router = APIRouter()


@router.post("/chat")
async def chat(req: ChatRequest):
    """
    SSE 流式聊天接口（核心接口）
    ============================
    接收用户问题，根据 graph_type 选择对应的工作流，以 SSE 方式实时推送执行过程。

    参数:
        req (ChatRequest): 聊天请求对象，包含以下字段：
            - question (str): 用户输入的问题文本
            - graph_type (str, 默认 "graph2"): 选择工作流
                * "graph1": 基础 RAG — Agent 判断 → 检索 → 文档评估 → 生成/重写
                * "graph2": Corrective RAG — 路由 → 检索/网络搜索 → 文档评估
                           → 生成 → 幻觉检测 → 回答评估 → 修正循环
            - thread_id (Optional[str], 默认 None): 会话唯一标识
                * None: 自动生成新的 UUID，开启新会话
                * 传入已有 ID: 继续之前的会话，保留历史对话上下文

    返回:
        EventSourceResponse: SSE 流式响应，事件类型包括：
            - session: 会话信息（thread_id）
            - node: 节点执行信息（当前哪个节点在处理）
            - answer: 生成的回答内容
            - done: 流结束标记
            - error: 错误信息

    流程:
        1. 判断 req.graph_type
           - 如果是 "graph1" → 走基础 RAG 管道
           - 否则（包括 "graph2" 或任何非 "graph1" 的值）→ 走 Corrective RAG 管道
        2. 将流式生成器包装为 EventSourceResponse 返回
           - 前端通过 EventSource API 连接此端点，实时接收推送
    """
    if req.graph_type == "graph1":
        # 条件成立：用户选择了基础 RAG 流程
        # → 调用 stream_graph1()，返回 Graph1 的 SSE 事件流
        return EventSourceResponse(stream_graph1(req))
    else:
        # 条件不成立：graph_type 不是 "graph1"（包括 "graph2" 或其他值）
        # → 默认走 Graph2（Corrective RAG），功能更完善
        return EventSourceResponse(stream_graph2(req))


@router.get("/graphs")
async def list_graphs():
    """
    获取可用的 RAG 工作流列表
    =========================
    供前端下拉菜单使用，让用户选择使用哪个工作流。

    返回格式:
        {
            "graphs": [
                {
                    "id": "graph1",
                    "name": "基础 RAG (Agent + 检索评估)",
                    "description": "Agent 判断是否需要检索 → 检索 → 文档评估 → 生成/重写"
                },
                {
                    "id": "graph2",
                    "name": "Corrective RAG (路由 + 幻觉检测)",
                    "description": "问题路由 → 检索/Web搜索 → 文档评估 → 生成 → 幻觉检测 → 答案评估"
                }
            ]
        }

    两个工作流的区别:
        - graph1: 简单直接，适合明确的知识库问答
        - graph2: 多轮修正，有幻觉检测和自动纠错，适合复杂问题
    """
    return {
        "graphs": [
            {
                "id": "graph1",
                "name": "基础 RAG (Agent + 检索评估)",
                "description": "Agent 判断是否需要检索 → 检索 → 文档评估 → 生成/重写",
            },
            {
                "id": "graph2",
                "name": "Corrective RAG (路由 + 幻觉检测)",
                "description": "问题路由 → 检索/Web搜索 → 文档评估 → 生成 → 幻觉检测 → 答案评估",
            },
        ]
    }
