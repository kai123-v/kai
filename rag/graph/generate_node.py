"""
Graph1 — 答案生成节点
=====================
当文档评估通过（文档与问题相关）后，此节点基于检索到的文档生成最终回答。

核心流程:
    检索文档 + 用户问题 → 提示词模板 → LLM → 字符串输出 → 返回
"""

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from rag.utils.message_utils import get_last_human_message
from rag.llm_models.embeddings_model import llm
from rag.utils.logger import log


def generate(state):
    """
    基于检索文档生成最终回答
    ========================
    将用户问题与检索到的文档组合成 RAG 提示词，让 LLM 基于文档生成答案。

    参数:
        state (AgentState):
            当前图状态，关键字段:
            - messages (List[BaseMessage]): 完整对话历史
              在这个节点被调用时，messages 列表的最后一条是检索工具返回的文档内容，
              更早的消息中包含用户原始问题。

    返回:
        dict: {"messages": [response]}
            response 是 LLM 生成的纯文本回答字符串。

    代码逻辑（5 步）:
        1. 从 messages 中提取用户问题（最后一条 HumanMessage）
        2. 从 messages 中提取检索文档（最后一条消息的 content）
        3. 构建 RAG 提示词模板，将问题与文档组合
        4. 构建处理链: prompt | llm | StrOutputParser
        5. 执行链，生成回答

    为什么用 StrOutputParser？
        LLM 默认返回 AIMessage 对象，包含 content + metadata。
        StrOutputParser 将其转换为纯字符串，方便后续处理。
        如果不用它，返回的 response 是 AIMessage 类型，也可以工作，
        但纯字符串更简洁。

    这个节点何时被调用？
        只有 grade_documents 返回 "generate" 时才会走到这里。
        也就是说，只有当检索到的文档被评估为"相关"时才生成答案。
        如果文档不相关，会先走 rewrite 节点重写问题。
    """
    log.info("生成答案")

    # ===== 1. 提取用户问题 =====
    messages = state["messages"]
    # 获取最后一条人类消息作为当前问题
    # 为什么不用 messages[0]？
    #   多轮对话中可能有多个 HumanMessage（用户追问、被重写的问题等）
    #   取"最后一条"确保拿到当前轮次的用户意图
    question = get_last_human_message(messages).content

    # ===== 2. 提取检索文档 =====
    # messages[-1]: 消息列表最后一条
    # 在 grade_documents 返回 "generate" 的流程中:
    #   agent → retrieve 后，最后一条是 ToolMessage，content 是检索到的文档内容
    # 所以这里直接取最后一条消息的 content 即可获得文档
    last_message = messages[-1]
    docs = last_message.content

    # ===== 3. 构建 RAG 提示词模板 =====
    # PromptTemplate: 定义提示词结构，占位符在实际调用时被替换
    # {question}: 用户问题占位符 → 替换为实际问题文本
    # {context}:  文档内容占位符 → 替换为检索到的文档文本
    prompt = PromptTemplate(
        template=(
            "你是一个问答任务助手。请根据以下检索到的上下文内容回答问题。"
            "如果不知道答案，请直接说明。回答保持简洁\n"
            "问题:{question}\n"
            "上下文:{context}"
        ),
        # input_variables: 告诉 LangChain 这个模板需要哪些变量
        # 调用 invoke() 时必须传入对应的键值对
        input_variables=["question", "context"],
    )

    # ===== 4. 构建处理链 =====
    # | (管道运算符) 是 LangChain 的 LCEL (LangChain Expression Language) 语法
    # prompt | llm | StrOutputParser() 的工作流:
    #   1. 将输入字典格式化为提示词文本
    #   2. 将提示词发送给 LLM
    #   3. LLM 返回 AIMessage
    #   4. StrOutputParser 提取 content 字段，返回纯字符串
    rag_chain = prompt | llm | StrOutputParser()

    # ===== 5. 执行链并返回 =====
    # invoke() 传入字典，键值对应 PromptTemplate 中定义的占位符
    response = rag_chain.invoke({"context": docs, "question": question})

    return {"messages": [AIMessage(content=response)]}
