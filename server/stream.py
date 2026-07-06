"""
SSE 流式推送模块
================
负责将 LangGraph 工作流的执行过程通过 Server-Sent Events (SSE) 实时推送给前端。

什么是 SSE (Server-Sent Events)？
    SSE 是一种服务器向客户端单向推送数据的技术，基于 HTTP 协议。
    客户端通过 EventSource API 连接后，服务器可以持续发送事件流。
    与 WebSocket 的区别：
    - SSE: 单向（服务器→客户端），基于 HTTP，更简单，自动重连
    - WebSocket: 双向通信，需要额外协议升级

SSE 事件类型说明：
    | 事件类型   | 携带数据                          | 触发时机           |
    |-----------|----------------------------------|-------------------|
    | session   | {thread_id}                      | 会话开始，告知会话ID |
    | node      | {node, label, detail}            | 每个节点执行完毕     |
    | answer    | {generation}                     | 生成回答完成        |
    | done      | {}                               | 整个流程结束        |
    | error     | {message}                        | 发生错误           |

stream_mode 说明：
    LangGraph 的 stream() 方法支持多种输出模式，通过 stream_mode 参数控制：
    - "values":  每步返回完整的状态快照 | 适合调试、看全貌
    - "updates": 每步只返回增量更新     | 适合追踪节点变化（本项目使用）
    - "messages": 只返回新消息          | 适合纯聊天流式输出
    - "debug":   详细调试信息           | 性能分析

    注意：stream_mode="updates" 返回的是节点原始输出，LangGraph 的 add_messages
    reducer 尚未执行。所以 Graph1 的 generate 返回 {"messages": [字符串]} 时，
    node_output["messages"][-1] 是原始 str 而非 AIMessage，不能访问 .content。
"""

import json
import uuid
from typing import AsyncGenerator

from server.schemas import ChatRequest


# ==================== 节点名称中英文映射 ====================
# 将 LangGraph 内部的英文节点名映射为前端展示的中文标签
NODE_LABELS = {
    "agent": "Agent 判断",               # Graph1: 智能体决定是否需要检索
    "retrieve": "文档检索",               # Graph1/2: 从 Milvus 向量库检索文档
    "rewrite": "问题重写",               # Graph1: 将问题改写为更易检索的形式
    "generate": "生成回答",              # Graph1/2: 基于检索结果生成最终回答
    "grade_documents": "文档评估",        # Graph2: 评估检索到的文档是否与问题相关
    "web_search": "网络搜索",            # Graph2: 通过百度 AI 搜索获取外部信息
    "transformer_query": "问题优化",      # Graph2: 优化问题表述
    "route_question": "问题路由",        # Graph2: 决定走向量检索还是网络搜索
}


def _make_event(event_type: str, data: dict) -> str:
    """
    构造 SSE 事件数据
    =================
    将事件类型和数据组装为符合 SSE 规范的 JSON 字符串。

    参数:
        event_type (str): 事件类型
            可选值: "session" | "node" | "answer" | "done" | "error"
            每种类型对应前端不同的处理逻辑
        data (dict): 事件携带的数据
            不同事件类型携带不同字段，详见各调用处。
            **data 把字典展开成键值对，等价于:
            {"type": event_type, "name": "张三", "age": 25}

    返回:
        str: SSE 格式的 JSON 字符串
        格式示例: '{"type": "node", "node": "retrieve", "label": "文档检索", "detail": "3 个文档"}'

    为什么用 ensure_ascii=False？
        确保中文不被转义为 unicode 编码（如 \\u6587\\u6863），
        前端可以直接读取中文，减少解码开销。
    """
    return json.dumps({"type": event_type, **data}, ensure_ascii=False)


def _extract_node_detail(node_name: str, node_output: dict) -> str:
    """
    从节点输出中提取可展示的摘要信息
    ================================
    不同节点输出不同的数据，此函数根据节点类型提取关键摘要，
    供前端在 NodeTimeline 中展示每个步骤的简要信息。

    参数:
        node_name (str): 节点内部名称
            可选值: "agent" | "retrieve" | "rewrite" | "generate" |
                   "web_search" | "grade_documents" | "transformer_query"
        node_output (dict): 节点执行后的输出数据
            Graph1: 以 messages 列表为主，如 {"messages": [AIMessage(...)]}
            Graph2: 包含独立字段，如 {"documents": [...], "question": "..."}

    返回:
        str: 人类可读的摘要信息
            - agent: "决定检索" 或 "直接回答: ..."
            - retrieve: "3 个文档"（Graph2）或 "1 条检索结果"（Graph1）
            - web_search: "搜索到 5 条结果"
            - grade_documents: "保留 2 个相关文档" 或 "无相关文档"
            - transformer_query: "重写为: 优化后的问题..."
            - rewrite: "重写为: 重写后的问题..."
            - generate: "生成 256 字"
            - 未知节点: ""（空字符串）
    """
    # --- Graph1 专属节点 ---

    if node_name == "agent":
        # 触发: agent 节点执行后
        # node_output = {"messages": [AIMessage]}
        #   AIMessage 有 tool_calls → Agent 决定检索
        #   AIMessage 无 tool_calls → Agent 直接回答（此时走流兜底，不经 generate）
        last_msg = node_output["messages"][-1]
        if getattr(last_msg, 'tool_calls', None):
            return "决定检索"
        return f"直接回答: {last_msg.content[:30]}"

    if node_name == "rewrite":
        # 触发: 检索文档被 grade_documents 判定不相关，Graph1 重写问题后重新交给 agent
        # node_output = {"messages": [AIMessage]}，内容是改写后的问题
        return f"重写为: {node_output['messages'][-1].content[:40]}"

    # --- Graph1 / Graph2 共用节点 ---

    if node_name == "retrieve":
        # Graph2: node_output = {"documents": [...], ...}  → 走 docs 分支
        # Graph1: ToolNode → node_output = {"messages": [ToolMessage]} → 走 messages 分支
        docs = node_output.get("documents", [])
        if docs:
            return f"{len(docs)} 个文档"
        return f"{len(node_output['messages'])} 条检索结果"

    if node_name == "generate":
        # Graph2: GraphState 有 generation 字段 → node_output["generation"] 存在
        # Graph1: AgentState 无 generation 字段 → 被 LangGraph 过滤，从 messages 取
        gen = node_output.get("generation", "") or node_output["messages"][-1]
        return f"生成 {len(gen)} 字"

    # --- Graph2 专属节点 ---

    if node_name == "web_search":
        # 触发: route_question 判定问题不在向量库范围，或 2 次优化后仍无相关文档
        return f"搜索到 {len(node_output['documents'])} 条结果"

    if node_name == "grade_documents":
        # 触发: 检索完成后，逐文档评估相关性
        #   docs 非空 → 有相关文档，进入 generate
        #   docs 为空 → 无相关文档，进入 transformer_query 或 web_search
        docs = node_output.get("documents", [])
        return f"保留 {len(docs)} 个相关文档" if docs else "无相关文档"

    if node_name == "transformer_query":
        # 触发: grade_documents 后无相关文档且 transforme_count < 2
        return f"重写为: {node_output['question'][:50]}"

    return ""


#返回一个异步生成器，每次yield 返回一个str的类型
async def stream_graph1(req: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Graph1（基础 RAG）SSE 流式处理
    ==============================
    使用 LangGraph 的 stream_mode="updates" 模式，精确追踪每个节点的执行。
    每完成一个节点就推送一个 "node" 事件，生成回答后推送 "answer" 事件。

    参数:
        req (ChatRequest): 聊天请求对象
            - question (str): 用户问题
            - graph_type (str): 应为 "graph1"
            - thread_id (Optional[str]): 会话唯一标识
                可选值:
                    - None: 不传入，后端自动生成新的 UUID，开启新会话
                    - 已有的 UUID 字符串: 传入之前对话的 thread_id，继续多轮对话

    完整流程（4 条路径）:
        路径1: agent 判断不需要检索 → agent 直接回复 → END
        路径2: agent → retrieve → grade(相关) → generate → END
        路径3: agent → retrieve → grade(不相关) → rewrite → agent 直接回复 → END
        路径4: agent → retrieve → grade(不相关) → rewrite → agent(检索) → retrieve → grade(相关) → generate → END

    回答提取策略:
        - 路径2/4 经过 generate 节点 → 从 node_output["messages"][-1] 取回答
        - 路径1/3 不经过 generate → 兜底从 graph.get_state() 取最后一条 AIMessage
          （图已通过 tools_condition → END，该消息必无 tool_calls）

    错误处理策略:
        - 服务商 API 返回 "sensitive" 错误 → 提示用户换个说法，不暴露原始错误
        - 其他错误 → 返回错误信息摘要（截断前 80 字符），避免前端显示过长
    """
    from rag.graph.graph1 import graph

    # ===== 1. 会话管理 =====
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    yield _make_event("session", {"thread_id": thread_id})
    yield _make_event("node", {"node": "user_input", "label": "用户提问"})

    # ===== 2. 构造输入 =====
    # Graph1 的状态只有一个 messages 字段
    # ("user", question) 是 LangChain 的消息简写格式，会自动转换为 HumanMessage
    inputs = {"messages": [("user", req.question)]}

    try:
        # ===== 3. 执行工作流 =====
        # graph.stream() 参数:
        #   inputs: 初始输入，放入图的 START 节点
        #   config: 配置对象，包含 thread_id 用于状态持久化
        #   stream_mode: "updates" — 只看增量，不看全貌
        # 返回值: 生成器，每次迭代是一个 {node_name: node_output} 字典
        # 特别说明: stream 只是返回一个生成器函数，不执行任何逻辑，
        #   只有当 for 循环调用时，才开始执行 LangGraph 图节点
        events = graph.stream(inputs, config, stream_mode="updates")
        last_generation = ""

        for event in events:
            for node_name, node_output in event.items():
                label = NODE_LABELS.get(node_name, node_name)
                detail = _extract_node_detail(node_name, node_output)

                yield _make_event("node", {
                    "node": node_name, "label": label, "detail": detail,
                })

                # generate 节点 → 提取回答
                # node_output = {"messages": [字符串]}，字符串是 StrOutputParser 的原始输出
                if node_name == "generate":
                    last_generation = node_output["messages"][-1]
                    yield _make_event("answer", {"generation": last_generation})

        # 兜底: agent 直接回复（路径1/3），未经过 generate 节点
        # 图已通过 tools_condition → END 结束，最后一条 AIMessage 必无 tool_calls
        if not last_generation:
            state = graph.get_state(config)
            for msg in reversed(state.values.get("messages", [])):
                if msg.type == "ai":
                    last_generation = msg.content
                    yield _make_event("answer", {"generation": last_generation})
                    break

    except Exception as e:
        error_msg = str(e)
        # 服务商 API 返回 "sensitive" → 内容审核拦截，提示用户换说法
        if "sensitive" in error_msg.lower():
            yield _make_event("error", {"message": "内容触发了安全审查，请换个说法重试"})
        else:
            yield _make_event("error", {"message": f"处理出错，请重试（{error_msg[:80]}）"})

    yield _make_event("done", {})


async def stream_graph2(req: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Graph2（Corrective RAG）SSE 流式处理
    ===================================
    与 stream_graph1 类似，但输入状态结构不同，包含更多字段。
    Graph2 的流程更复杂：路由 → 检索/网络搜索 → 文档评估 → 生成 → 幻觉检测 → 回答评估。

    与 stream_graph1 的关键区别:
        1. 输入状态: Graph1 只有 messages，Graph2 有 question + documents + generation
           + transforme_count + messages 五个字段
        2. 路由机制: Graph2 起始就有 route_question 判断（向量检索 vs 网络搜索）
        3. 幻觉检测: Graph2 会检查生成内容是否基于检索文档，如果不基于则重新生成
        4. 循环计数: transforme_count 限制问题优化最多 2 次，防止死循环，
           超过后降级为网络搜索

    参数:
        req (ChatRequest): 聊天请求对象
            - question (str): 用户问题
            - graph_type (str): 应为 "graph2"
            - thread_id (Optional[str]): 会话唯一标识

    每条路径最终都经过 generate 节点（GraphState 有 generation 字段，不会被过滤），
    所以不需要 Graph1 那样的兜底逻辑。
    """
    from rag.graph2.graph_2 import graph

    # ===== 1. 会话管理 =====
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    yield _make_event("session", {"thread_id": thread_id})
    yield _make_event("node", {"node": "user_input", "label": "用户提问"})

    # ===== 2. 构造 Graph2 的输入状态 =====
    # 5 个字段各有明确语义，与 AgentState（只有 messages）不同:
    #   question: 当前处理的问题（可能被 transformer_query 优化）
    #   documents: 检索到的文档列表（可能被 grade_documents 过滤）
    #   generation: LLM 生成的回答（可能被 grade_hallucinations 评估后重试）
    #   transforme_count: 问题优化次数计数器（防止无限循环，最多 2 次）
    #   messages: 对话历史（支持多轮对话记忆）
    #   将上述字段传入给graph的state
    inputs = {
        "question": req.question,
        "documents": [],
        "generation": "",
        "transforme_count": 0,
        "messages": [("user", req.question)],
    }

    try:
        # ===== 3. 执行工作流 =====
        events = graph.stream(inputs, config, stream_mode="updates")
        last_generation = ""

        for event in events:
            for node_name, node_output in event.items():
                label = NODE_LABELS.get(node_name, node_name)
                detail = _extract_node_detail(node_name, node_output)

                yield _make_event("node", {
                    "node": node_name, "label": label, "detail": detail,
                })

                # GraphState 有 generation 字段，generate 节点始终返回非空字符串
                if node_name == "generate":
                    last_generation = node_output["generation"]
                    yield _make_event("answer", {"generation": last_generation})

    except Exception as e:
        error_msg = str(e)
        if "sensitive" in error_msg.lower():
            yield _make_event("error", {"message": "内容触发了安全审查，请换个说法重试"})
        else:
            yield _make_event("error", {"message": f"处理出错，请重试（{error_msg[:80]}）"})

    yield _make_event("done", {})
