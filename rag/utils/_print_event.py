"""
事件打印工具模块
================
用于在终端中格式化打印 LangGraph 执行过程中的事件信息。
主要用于开发调试，在 stream.py 中也有类似的（但通过 SSE 推送给前端）。

两个函数的区别:
    print_event:         用于 Graph1（状态结构以 messages 为主）
    print_event_graph2:  用于 Graph2（状态结构包含 question/generation/documents/messages）

去重机制:
    使用 _printed 集合记录已打印的消息/内容 ID。
    避免在多轮对话或重试循环中重复打印相同内容。
    _printed 是一个 set，O(1) 的查找和插入效率。
"""


def print_event(event: dict, _printed: set, max_length=1500):
    """
    打印 Graph1 事件信息
    ====================
    适配 Graph1 的 AgentState 结构（以 messages 列表为核心）。

    参数:
        event (dict):
            LangGraph stream() 返回的单个事件。
            Graph1 的 stream_mode="values" 时，event 包含完整的 messages 列表。
            结构示例:
            {
                "dialog_state": [...],
                "messages": [HumanMessage(...), AIMessage(...), ...]
            }

        _printed (set):
            已打印消息 ID 的集合。
            用于去重：相同 ID 的消息只打印一次。
            注意: 这是一个可变集合，函数会修改它（添加新打印的消息 ID）。

        max_length (int, 默认 1500):
            消息内容的最大显示长度。
            超过此长度的消息会被截断，末尾追加 "...(已截断）"。
            为什么需要截断？
                - 检索文档可能非常长（数千字符）
                - 在终端中完整打印会刷屏，影响可读性
                - 1500 字符提供了足够的预览又能保持输出整洁
            可选值:
                - 500: 简洁预览
                - 1500: 平衡（默认）
                - 0 或负数: 不截断（不推荐，终端可能卡顿）

    打印逻辑:
        1. 如果 event 包含 "dialog_state" → 打印当前对话状态
        2. 如果 event 包含 "messages" → 取最后一条消息
        3. 如果该消息 ID 未打印过 → 格式化并打印
        4. 如果内容超长 → 截断处理
        5. 将消息 ID 加入 _printed（去重记录）
    """
    # 打印对话状态（如果有）
    current_state = event.get("dialog_state")
    if current_state:
        # current_state 是列表，[-1] 取最后一个状态（当前状态）
        print("当前处于:", current_state[-1])

    # 打印最新消息
    message = event.get("messages")
    if message:
        # messages 是列表，[-1] 取最后一条消息
        message = message[-1]

        # 去重检查: 如果消息 ID 已在 _printed 中 → 跳过
        if message.id not in _printed:
            # pretty_repr(html=True): LangChain 的格式化输出方法
            # 以人类友好的方式展示消息内容，html=True 使用富文本格式
            msg_repr = message.pretty_repr(html=True)

            # 长度截断
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + "...(已截断）"

            print(msg_repr)
            _printed.add(message.id)  # 记录已打印


def print_event_graph2(event: dict, _printed: set, max_length=1500):
    """
    打印 Graph2 事件信息
    ====================
    适配 Graph2 的 GraphState 结构（包含 question/generation/documents/messages）。

    与 print_event 的核心区别:
        Graph2 的状态不只是 messages，还有独立的 question、generation、documents 字段。
        本函数分别处理这些字段，提供更结构化的输出。

    参数:
        event (dict):
            Graph2 stream() 事件，stream_mode="values" 时包含完整状态。
            结构示例:
            {
                "question": "什么是机器学习？",
                "generation": "机器学习是...",
                "documents": [Document(...), Document(...)],
                "messages": [HumanMessage(...)]
            }

        _printed (set):
            去重集合。与 print_event 不同，这里存储的是内容文本（而非消息 ID）。
            因为 Graph2 的 question/generation 没有 .id 属性。

        max_length (int, 默认 1500):
            最大显示长度。

    打印顺序:
        1. question   — 用户问题（优先打印，因为是对话起点）
        2. documents  — 文档数量（统计信息）
        3. generation — AI 回答（核心输出）
        4. messages   — 最新消息（兼容格式）
    """
    question = event.get("question")
    generation = event.get("generation")
    documents = event.get("documents")
    messages = event.get("messages")

    # ===== 1. 打印问题 =====
    if question and question not in _printed:
        print(f"问题: {question}")
        _printed.add(question)  # 记录问题文本本身

    # ===== 2. 打印文档数量 =====
    if documents is not None:
        doc_count = len(documents) if isinstance(documents, list) else 1
        print(f"检索到 {doc_count} 个文档")

    # ===== 3. 打印生成结果 =====
    if generation and generation not in _printed:
        # 截断过长的回答
        if len(generation) > max_length:
            generation = generation[:max_length] + "...(已截断）"
        print(f"回答: {generation}")
        _printed.add(generation)

    # ===== 4. 打印最新消息（兼容形式） =====
    if messages:
        last_msg = messages[-1]
        # 使用消息 ID 去重（与 print_event 一致）
        if last_msg.id not in _printed:
            msg_repr = last_msg.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + "...(已截断）"
            print(msg_repr)
            _printed.add(last_msg.id)
