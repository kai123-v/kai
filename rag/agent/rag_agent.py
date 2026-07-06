"""
RAG Agent 定义模块（独立版）
============================
使用 LangChain 的 create_tool_calling_agent 和 AgentExecutor 创建一个
带工具调用和会话记忆的完整 Agent。

注意: 这个 Agent 是独立运行的，**不直接用于 Graph1 工作流**。
       Graph1 在 agent_node.py 中直接使用 llm.bind_tools() + model.invoke()
       的底层方式，将工具调用决策交给 LangGraph 的图机制来控制。
       本文件更多是演示 LangChain Agent 的标准用法，或作为独立对话机器人使用。

Agent 的核心概念:
    1. Agent (智能体): 能使用工具的 LLM
       它不只是生成文本，还能决定"应该调用哪个工具"、"传什么参数"
    2. Tool (工具): Agent 可以调用的函数
       本项目: retriever_tool — 从向量库检索文档
    3. AgentExecutor (执行器): 管理 Agent 的"思考-行动"循环
       循环: 调用 LLM → 如果需要工具 → 执行工具 → 将结果返回 LLM → 再判断
       直到 LLM 认为不需要更多工具，生成最终回答

   与 Graph1 中 Agent 的关系:
       这是一个**完整的独立 Agent**（自带工具调用循环），
       而 Graph1 中的 agent_node 只是**一个节点**（工具调用循环由 Graph 管理）。
       两者的层次不同:
       - 本文件: Agent 层（高层封装，开箱即用）
       - agent_node: 节点层（底层控制，与图编排配合）

记忆功能 (Memory):
    RunnableWithMessageHistory 为 Agent 增加了对话记忆能力:
    - 每次对话的 messages 会根据 session_id 保存
    - 下次相同 session_id 的对话会加载历史消息
    - 存储: 内存中的 store 字典（重启丢失）
"""

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

from langchain_community.chat_message_histories import ChatMessageHistory

from rag.llm_models.embeddings_model import llm
from rag.tools.retriever import retriever_tool


# ==================== 1. 提示词模板 ====================

# ChatPromptTemplate.from_messages: 构建多角色对话格式的提示词
prompt = ChatPromptTemplate.from_messages(
    [
        # 系统消息: 定义 AI 助手的角色和行为
        ("system", "你是一个智能助手，尽可能的调用工具回答用户的问题"),

        # 历史对话占位符 (MessagesPlaceholder)
        # variable_name="chat_history": 占位符的名称，运行时用实际历史替换
        # optional=True: 首次对话时可以不传入（没有历史记录）
        # 这是 LangChain 标准的历史记录注入方式
        MessagesPlaceholder(variable_name="chat_history", optional=True),

        # 用户输入占位符
        # "{input}": 用户在本次对话中输入的问题
        ("human", "{input}"),

        # 工具调用记录占位符
        # variable_name="agent_scratchpad": Agent 的工具调用中间步骤
        # optional=True: 首次调用 LLM 时还没有工具调用记录
        # AgentExecutor 自动管理此占位符，不需要手动传入
        MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
    ]
)


# ==================== 2. 创建 Agent ====================

# create_tool_calling_agent: LangChain 提供的函数，创建支持工具调用的 Agent
# 参数:
#   llm: 大语言模型实例
#   [retriever_tool]: Agent 可用的工具列表
#      这里只有一个检索工具，但可以是多个
#      Agent 会根据用户问题自行选择调用哪个（或哪些）工具
#   prompt: 提示词模板（包含 chat_history、input、agent_scratchpad）
agent = create_tool_calling_agent(llm, [retriever_tool], prompt)


# ==================== 3. 测试: 独立 Agent（无记忆）====================

if __name__ == "__main__":
    """
    测试 1: 无记忆的 Agent

    这个测试展示了最基本的 Agent 使用方式:
    1. 创建 AgentExecutor
    2. 调用 invoke() 传入用户问题
    3. Agent 自动判断是否需要工具 → 调用工具 → 生成回答

    AgentExecutor 的作用：
    管理 Agent 的工具调用循环 (Thought-Action-Observation 循环):
    - Thought: LLM 分析问题，决定调用哪个工具，传什么参数
    - Action: Executor 执行工具调用（如调用 retriever_tool）
    - Observation: 将工具返回结果交给 LLM
    - 重复以上步骤，直到 LLM 决定不再需要工具
    - 最终 LLM 生成自然语言回答
    """
    executor = AgentExecutor(
        agent=agent,
        tools=[retriever_tool],  # 执行器可用的工具
    )
    res = executor.invoke({"input": "什么是机器学习"})
    print(res)


# ==================== 4. 会话记忆 ====================

# store 字典: 存储所有会话的历史消息
# 键: session_id（会话唯一标识）
# 值: ChatMessageHistory 对象（该会话的消息列表）
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    获取或创建会话的历史消息容器
    ============================
    由 RunnableWithMessageHistory 自动调用。
    每次对话时，LangChain 传入 config 中的 session_id，
    调用此函数获取该会话的历史消息。

    参数:
        session_id (str):
            会话唯一标识。由调用方通过 config["configurable"]["session_id"] 传入。
            例如: "zs123"、"user_abc" 等（可以使用任何字符串作为 ID）

    返回:
        BaseChatMessageHistory: 该会话的消息历史容器

    代码逻辑:
        1. 检查 store 字典中是否已有该 session_id
        2. 如果有 → 返回已有的 ChatMessageHistory（包含历史消息）
        3. 如果没有 → 创建新的 ChatMessageHistory（空的历史）
        - ChatMessageHistory() 内部是一个消息列表，append 新消息
        - 它是 LangChain 提供的消息储存容器

    为什么用内存字典存储？
        - 简单直接，开发/测试时使用
        - 缺点: 重启后所有历史丢失
        生产环境替代方案:
        - RedisChatMessageHistory: 存储在 Redis 中（推荐）
        - SQLChatMessageHistory: 存储在数据库中
        - FileChatMessageHistory: 存储在文件中
    """
    # messages = [
    #     {"role": "system", "content": "你是一个专业的AI助手，请用中文回答问题。"},
    #     {"role": "user", "content": "帮我总结一下AI Agent的特点。"},
    #     {"role": "assistant", "content": "AI Agent的核心特点是...（完整回复）"},
    #     {"role": "user", "content": "那它和普通LLM有什么区别？"},
    #     # ... 后续消息
    # ] 这个就是历史会话记录
    # store = {
    #     "user_1": [
    #         {"role": "human", "content": "你好"},
    #         {"role": "ai", "content": "你好！"},
    #         {"role": "human", "content": "我叫小明"},
    #         {"role": "ai", "content": "你好小明！"}
    #     ]
    # }
    if session_id not in store:
        # 该会话第一次对话 → 创建新的历史容器 ChatMessageHistory()用于存储和管理多轮对话历史的一个类
        store[session_id] = ChatMessageHistory()
    # 返回已有（或刚创建）的历史容器
    return store[session_id]


# ==================== 5. 测试: 带记忆的 Agent ====================

if __name__ == "__main__":
    # RunnableWithMessageHistory: 为 Agent 添加会话记忆功能
    # 工作原理:
    #   1. 执行前: 通过 get_session_history 加载历史消息 → 注入到 chat_history 占位符
    #   2. 执行中: Agent 正常处理（工具调用 + 生成回答）
    #   3. 执行后: 自动将本轮的 input 和 output 追加到 ChatMessageHistory
    agent_with_history = RunnableWithMessageHistory(
        executor,                 # 要包装的执行器
        get_session_history,      # 历史获取函数
        input_messages_key="input",       # 输入消息的键名（对应 prompt 中的 {input}）
        history_messages_key="chat_history",  # 历史消息的键名（对应 prompt 中的 {chat_history}）
    )

    # 第一次调用（session_id="zs123" 的第一次对话）
    res2 = agent_with_history.invoke(
        {"input": "什么是光刻机"},
        # configurable.session_id: 指定会话 ID
        # 同一 session_id 的后续调用会共享历史
        config={"configurable": {"session_id": "zs123"}},
    )
    print(res2)

    # 如果继续调用（同一 session_id）:
    # res3 = agent_with_history.invoke(
    #     {"input": "它用在什么领域？"},  # "它" 指代"光刻机"，AI 需要历史记忆才能理解
    #     config={"configurable": {"session_id": "zs123"}},
    # )
    # AI 因为记得之前讨论了"光刻机"，所以能正确理解"它"的指代

    #带记忆功能。编写写历史会话消息记录获取函数。然后用RunnableWithMessageHistory()这个类。把这个函数跟executor传进去。
    #agentexcutor执行agent的管理器。传入建好的agent跟工具。两者都要邦工具。agent绑了是为了知道有什么工具。excutor是为了知道怎么调用，怎么执行
    #langgraph走到工具节点会自己执行。有自己控制的流。能走。独立的agent需要executor帮他循环跟执行工具。