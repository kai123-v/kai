"""
Graph1 — Agent 决策节点
=======================
这个节点是整个 Graph1 的"大脑"。它接收当前对话状态，
由 LLM 判断是否需要调用检索工具，还是直接回答/结束对话。

核心机制：LLM 的 Tool Calling (Function Calling)
    1. 将可用工具列表"绑定"到 LLM
    2. LLM 根据上下文判断是否需要调用工具
    3. 如果需要：
       → AIMessage 中包含 tool_calls 字段，指定要调用的工具名和参数
       → 后续由 tools_condition 路由到 retrieve 节点执行实际检索
    4. 如果不需要：
       → AIMessage 中不包含 tool_calls，直接包含文本回复
       → 后续由 tools_condition 路由到 END，结束工作流

和 rag/agent/rag_agent.py 的关系:
    rag_agent.py 使用 create_tool_calling_agent() 创建完整的 Agent+Executor 组合，
    包含自动的工具调用循环。但本文件直接使用 llm.bind_tools() + model.invoke()
    的更底层方式，让 LangGraph 的图机制来控制工具调用流程。
    这种方式给予了更精细的流程控制。
"""

from rag.llm_models.embeddings_model import llm
from rag.tools.retriever import retriever_tool
from rag.utils.logger import log


def agent(state):
    """
    智能体决策节点
    ==============
    接收当前对话状态，由 LLM 判断下一步动作（检索 或 直接回答）。

    参数:
        state (AgentState):
            当前图状态，关键字段:
            - messages (List[BaseMessage]): 完整对话历史
              包含: HumanMessage（用户输入） + AIMessage（AI 回复） + ToolMessage（工具结果）
              LLM 会根据整个历史来判断当前需要做什么。

    返回:
        dict: 更新后的状态字典
            {"messages": [AIMessage]}
            LangGraph 的 add_messages reducer 会将返回的 AIMessage 追加到 messages 列表。

    LLM 的决策逻辑（由模型自主决定，不由代码控制）:
        1. LLM 分析 messages 中的最后一条用户消息
        2. 判断是否需要检索知识库:
           需要检索的情况:
             - 问题涉及 AI、机器学习、深度学习、数学等专业领域
             - 问题需要查证事实
             - 问题与向量库中的知识高度相关
           不需要检索的情况:
             - 简单的寒暄（你好、谢谢）
             - 纯闲聊话题
             - LLM 自身知识足以回答的通用问题
        3. 如果需要检索 → 返回包含 tool_calls 的 AIMessage
           不需要检索 → 返回普通 AIMessage（直接回答或拒绝回答）

    绑定工具 vs 创建 Agent 的区别:
        - llm.bind_tools([tool]): 只是告诉 LLM"你可以用这个工具"，
           但不会自动调用。工具调用由后续的 ToolNode 或 tools_condition 处理。
        - create_tool_calling_agent(): 创建完整的 Agent，包含自动的工具调用循环，
           Agent 会自动迭代"调用工具 → 获取结果 → 再判断"直到不需要更多工具。

    为什么 Graph1 只用 bind_tools？
        因为 Graph1 把"工具调用循环"交给了 LangGraph 的图机制:
        agent(判断) → retrieve(执行) → grade_documents(评估) → generate/rewrite
        每个步骤都是独立的节点，给了更精细的控制能力。
    """
    log.info("调用智能体")

    # ===== 1. 获取消息历史 =====
    # state["messages"] 包含完整的对话历史
    # 例如:
    #   [HumanMessage("什么是机器学习？")]
    # 或多次迭代后:
    #   [HumanMessage("什么是机器学习？"),
    #    AIMessage(tool_calls=[...]),        # LLM 决定调用检索
    #    ToolMessage(content="文档内容..."),  # 检索结果
    #    AIMessage("机器学习是...")]          # 之前生成的回答
    messages = state["messages"]

    # ===== 2. 绑定工具到 LLM =====
    # bind_tools() 将工具定义注入到 LLM 的上下文中
    # LLM 会知道有一个叫 "rag_retriever" 的工具可用，
    # 当用户问题需要检索知识库时，LLM 会在 AIMessage 中设置 tool_calls
    #
    # 参数: [retriever_tool] — 可用工具列表
    #   这里只有一个检索工具，但可以是多个（如果还有计算器、翻译等工具）
    # retriever_tool 的定义见: rag/tools/retriever.py
    model = llm.bind_tools([retriever_tool])

    # ===== 3. 调用 LLM =====
    # invoke() 是同步调用
    # LLM 会根据完整 messages 历史做出判断
    #
    # 可能的返回结果:
    #   情况1 — 需要检索:  这个结构是langchain-api底层框架用的消息结构。
    #     AIMessage(
    #       content="",                       # 内容通常为空
    #       tool_calls=[{                     # 包含工具调用请求
    #         "name": "rag_retriever",         # 工具名
    #         "args": {"query": "机器学习"},    # 工具参数（通常与用户问题相同或相近）
    #         "id": "call_xxx"                # 调用 ID
    #       }]
    #     )
    #   情况2 — 不需要检索（直接回答）:
    #     AIMessage(content="你好！有什么可以帮助你的吗？")
    response = model.invoke(messages)

    # ===== 4. 返回更新 =====
    # 返回列表形式，因为 add_messages reducer 期望接收一个列表
    # 如果返回 {"messages": response}（非列表），
    #   add_messages 也能处理单个消息，但用列表更明确
    return {"messages": [response]}
