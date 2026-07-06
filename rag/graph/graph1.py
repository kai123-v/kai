"""
Graph1 — 基础 RAG 工作流编排
=============================
这是第一个 RAG 管道的主流程文件，负责：
1. 定义所有节点（agent, retrieve, rewrite, generate）
2. 定义所有边（固定边 + 条件边）
3. 编译工作流图
4. 配置记忆功能

Graph1 完整流程:
    START
      │
      ▼
    ┌──────┐
    │agent │  ← 智能体根据问题决定是否需要检索
    └──┬───┘
       │
       ├── 不需要检索 ──▶ END（直接回答或拒绝回答）
       │
       └── 需要检索
              │
              ▼
         ┌──────────┐
         │ retrieve  │  ← 从 Milvus 向量库检索相关文档
         └────┬──────┘
              │
              ▼
      ┌─────────────────┐
      │ grade_documents  │  ← 评估检索到的文档是否与问题相关
      └───┬─────────┬───┘
          │         │
          │ 相关    │ 不相关
          ▼         ▼
     ┌──────────┐  ┌─────────┐
     │ generate │  │ rewrite │  ← 重写问题，让检索更容易
     └────┬─────┘  └────┬────┘
          │              │
          ▼              │
         END             └──→ 回到 agent（用新问题重新决策）

关键设计决策:
    1. 为什么 retriever 节点使用 ToolNode 而非自定义函数？
       → ToolNode 是 LangGraph 预置的节点，自动处理工具调用的输入/输出格式。
          它从 AIMessage.tool_calls 中提取参数，执行工具，返回 ToolMessage。
          使用 ToolNode 可以省去手写格式转换代码。

    2. 为什么用 tools_condition 条件边？
       → tools_condition 检查 AIMessage 中是否包含 tool_calls：
         - 有 tool_calls → 走 "tools" 分支（去 retrieve 节点）
         - 无 tool_calls → 走 END 分支（Agent 认为不需要检索，直接结束）

    3. 什么是 MemorySaver？
       → LangGraph 的检查点 (checkpoint) 机制。每次节点执行后自动保存状态。
         重启服务后内存中的数据会丢失，生产环境建议换为 RedisSaver 或 SqliteSaver。
"""

import uuid
from typing import Literal, List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from rag.graph.agent_node import agent
from rag.graph.generate_node import generate
from rag.graph.graph_state import AgentState, Grade
from rag.graph.rewrite_node import rewrite
from rag.llm_models.embeddings_model import llm
from rag.tools.retriever import retriever_tool
from rag.utils import _print_event
from rag.utils._print_event import print_event
from rag.utils.logger import log
from rag.utils.message_utils import get_last_human_message


# ==================== 条件路由函数 ====================

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    文档相关性评估（条件路由函数）
    ============================
    判断检索到的文档是否与用户问题相关，决定下一步是生成答案还是重写问题。

    这个函数是一个"动态路由"，LangGraph 在运行时调用它来决定走哪条边。
    它和 tools_condition 的作用相同，只是逻辑是自己写的。

    参数:
        state (AgentState):
            当前图状态，包含完整的 messages 列表。
            我们从中提取:
            - 最后一条消息（检索工具返回的文档内容）
            - 最后一条人类消息（用户原始问题）

    返回:
        Literal["generate", "rewrite"]:
            路由目标节点，必须是已定义的节点名称。
            - "generate": 文档相关 → 直接生成答案
            - "rewrite":  文档不相关 → 重写问题后重新检索

    详细代码逻辑:
        1. 获取带结构化输出的 LLM（要求按 Grade 格式输出 yes/no）
        2. 构建评分提示词模板，包含文档内容和用户问题
        3. 从 messages 中提取：
           - question: 最后一条人类消息（用户原始问题）
           - docs: messages[-1] 的内容（即检索工具最后一次返回的文档内容）
        4. 调用评分链，LLM 返回 Grade 对象
        5. 读取 binary_score：
           - 如果是 "yes" → 返回 "generate"（文档相关，可以生成答案）
           - 否则（"no" 或其它值）→ 返回 "rewrite"（文档不相关，需要重写问题）

    为什么不直接生成答案而要评估？
        如果检索到的文档完全不相关问题，生成答案会产生"幻觉"（编造内容）。
        通过评估文档相关性，可以过滤掉这种情况，提高答案质量。
    """
    log.info("--检查document的相关性")

    # ===== 1. 创建带结构化输出的 LLM =====
    # with_structured_output() 让 LLM 不输出自然语言，而是输出指定的 Pydantic 模型
    # method="function_calling": 使用 OpenAI 兼容的 function calling 机制
    #   → LLM 不生成文本，而是调用"函数"来返回结构化数据
    #   可选 method 值:
    #     - "function_calling": 使用 function/tool calling（最常用，OpenAI 兼容）
    #     - "json_mode": 使用 JSON mode（强制 LLM 输出纯 JSON）
    #     - "json_schema": 使用结构化 JSON schema（更严格的结构约束）
    llm_with_structured = llm.with_structured_output(Grade, method="function_calling")

    # ===== 2. 构建评分提示词 =====
    prompt = PromptTemplate(
        template="""你是一个评估检索文档与用户问题相关性的评分器。\n
                    这是检索到的文档:\n\n{context}\n\n
                    这是用户的问题：{question}\n
                    如果文档包含与用户相关的关键词或语义含义,则评为相关。\n
                    给出二元评分'yes' 或'no'来表示文档是否与问题相关。
                """,
        # input_variables: 提示词中的占位符列表
        # {context} → 替换为检索到的文档内容
        # {question} → 替换为用户原始问题
        input_variables=["context", "question"],
    )

    # ===== 3. 构建处理链 =====
    # | 是 LangChain 的管道运算符（LCEL语法）:
    #   prompt | llm_with_structured
    #   等价于:
    #   1. 将输入格式化为提示词
    #   2. 发送给 LLM
    #   3. 返回 LLM 的结构化输出（Grade 对象）
    chain = prompt | llm_with_structured

    # ===== 4. 提取上下文数据 =====
    messages = state["messages"]
    # messages[-1]: 消息列表最后一条
    #   在 agent → retrieve 流程后，最后一条是检索工具返回的 ToolMessage
    #   其 content 字段包含检索到的文档内容
    last_message = messages[-1]

    # 获取原始用户问题
    # 为什么用 get_last_human_message 而非直接取 messages[0]？
    #   在多轮对话中，messages 可能有多个人类消息（用户追问、重写的问题等）。
    #   取"最后一条"人类消息才能获取当前轮次的用户意图。
    question = get_last_human_message(messages).content

    # docs: 检索工具返回的文档内容字符串
    docs = last_message.content

    # ===== 5. 执行评分 =====
    scored_result: Grade = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score

    # ===== 6. 路由判断 =====
    if score == "yes":
        # 条件成立: 文档与问题相关
        # → 走向 generate 节点，基于这些文档生成答案
        print("输出：文档相关")
        return "generate"
    else:
        # 条件不成立: 文档与问题不相关
        # → 走向 rewrite 节点，重写问题后重新检索
        # 可能原因: 检索到的文档与问题语义不匹配、关键词语义鸿沟等
        print("--输出：文档不相关--")
        print(score)
        return "rewrite"


# ==================== 构建工作流图 ====================

# 创建状态图，指定状态类型为 AgentState
# StateGraph 是 LangGraph 的核心类，用于定义有向图结构
workflow = StateGraph(AgentState)

# ===== 添加节点 =====
# 每个节点是一个可调用对象（函数、Runnable 等），接收 state 并返回更新后的 state 字典

# agent 节点: 智能体决策，判断是否需要检索工具
# 来源: rag/graph/agent_node.py 中的 agent() 函数
workflow.add_node("agent", agent)

# retrieve 节点: 文档检索
# ToolNode 是 LangGraph 内置的工具节点，自动处理:
#   1. 从 AIMessage.tool_calls 中提取工具调用参数
#   2. 执行工具函数
#   3. 将执行结果包装为 ToolMessage
#   4. 将 ToolMessage 追加到 messages 列表中
# 参数: [retriever_tool] → 该节点可用的工具列表（这里只有一个检索工具）
workflow.add_node("retrieve", ToolNode([retriever_tool]))

# rewrite 节点: 问题重写，优化查询表述
workflow.add_node("rewrite", rewrite)

# generate 节点: 答案生成，基于检索文档生成最终回答
workflow.add_node("generate", generate)


# ===== 添加边 =====
# 边定义了节点之间的执行顺序。LangGraph 支持三种边:
#   - 固定边 (add_edge): 始终从 A 到 B
#   - 条件边 (add_conditional_edges): 根据函数返回值决定走向哪个节点

# 边 1: START → agent（固定边）
# 工作流始终从 agent 节点开始
workflow.add_edge(START, "agent")

# 边 2: agent → ?（条件边，使用 LangGraph 内置的 tools_condition）
# tools_condition 的工作原理:
#   1. 检查 messages 列表最后一条 AIMessage
#   2. 如果 AIMessage 包含 tool_calls（说明 LLM 想调用工具）
#      → 走 "tools" 分支 → 进入 retrieve 节点
#   3. 如果 AIMessage 不包含 tool_calls（说明 LLM 给出了直接回答）
#      → 走 END 分支 → 工作流结束
# 这是 LangGraph 处理 Agent 工具调用的标准模式
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",  # Agent 决定使用工具 → 去检索
        END: END,            # Agent 决定不需要工具 → 结束
    },
)

# 边 3: retrieve → ?（条件边，使用自定义的 grade_documents 函数）
# grade_documents 是自己的路由函数，已经通过返回值的 Literal 类型写死了返回路径
# 所以这里不需要再提供映射字典，LangGraph 直接用函数返回值作为目标节点名
# 等价于:
#   grade_documents 返回 "generate" → 去 generate 节点
#   grade_documents 返回 "rewrite"  → 去 rewrite 节点
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    # 注意: 这里没有映射字典！
    # 因为 Python 的 Literal 类型已经定义了返回值只能是 "generate" 或 "rewrite"
    # LangGraph 自动将返回值用作目标节点名
)

# 边 4: rewrite → agent（固定边）
# 重写问题后回到 agent，用新的问题表述重新决策
# 形成循环: agent → retrieve → grade → rewrite → agent
workflow.add_edge("rewrite", "agent")

# 边 5: generate → END（固定边）
# 生成答案后工作流结束
workflow.add_edge("generate", END)


# ==================== 记忆功能配置 ====================

# MemorySaver: LangGraph 的内存检查点存储
# 作用: 每次节点执行后自动保存完整状态，下次执行时自动恢复
# 存储内容: 包括 messages 列表、图位置、配置等
#
# 存储位置选项:
#   - MemorySaver(): 保存在内存中（当前使用）
#     优点: 速度快，无需额外依赖
#     缺点: 服务重启后所有对话历史丢失
#     适用: 开发调试
#   - SqliteSaver.from_conn_string("checkpoints.sqlite"): 保存在 SQLite 中
#     优点: 持久化存储，重启不丢失
#     适用: 小规模生产
#   - RedisSaver(redis_client): 保存在 Redis 中
#     优点: 高性能，支持分布式
#     适用: 大规模生产
memory = MemorySaver()

# compile: 编译工作流图
# checkpointer=memory: 启用检查点机制，实现状态持久化
# 编译后的 graph 对象可以直接调用 invoke()、stream()、astream() 等方法
graph = workflow.compile(checkpointer=memory)


# ==================== 会话配置 ====================

# config 用于 graph.invoke/stream 时传递运行时配置
# configurable.thread_id:
#   - LangGraph 用此 ID 作为检查点的 key
#   - 不同 thread_id → 不同的对话历史（独立会话）
#   - 相同 thread_id → 共享对话历史（同一会话的多轮对话）
#   - 值: uuid.uuid4() 生成全局唯一标识符
config = {
    "configurable": {
        "thread_id": str(uuid.uuid4())
    }
}


# ==================== 交互式测试入口 ====================

if __name__ == "__main__":
    """
    命令行交互式测试
    ================
    在终端中直接运行此文件可以进行对话测试。
    输入问题 → 查看工作流执行过程 → 获取回答。
    输入 q / exit / quit 退出。

    stream_mode 可选值对比:
        - "values":  每一步返回完整状态快照 | 输出量大，包含所有历史消息
        - "updates": 只返回增量更新         | 输出精简，只显示新产生的消息
        - "messages": 只返回新消息的 token 流 | 适合流式聊天应用
        - "debug":   包含详细的调试信息      | 适合性能分析和问题排查
    """
    _printed = set()  # 用于去重：避免同一消息在同一轮对话中重复打印
    while True:
        question = input("用户:")
        # 退出条件：输入 q、exit 或 quit
        if question.lower() in ["q", "exit", "quit"]:
            print("对话结束，拜拜")
            break
        else:
            # 构造输入：("user", question) 是 LangChain 的消息简写
            inputs = {"messages": [("user", question)]}

            # 执行工作流
            # stream_mode="values": 每一步返回完整状态
            # 这样每个 event 都能看到完整的 messages 列表变化
            events = graph.stream(inputs, config, stream_mode="values")

            # 遍历事件并打印
            for event in events:
                print_event(event, _printed)
