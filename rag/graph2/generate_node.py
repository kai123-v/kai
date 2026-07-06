"""
Graph2 — 答案生成节点
=====================
基于检索到的文档（向量库或网络搜索）生成最终回答。

与 Graph1 generate_node 的重要区别:
    1. 支持对话历史: 将最近 6 条对话消息压缩为上下文，让 LLM 理解对话背景
    2. 支持幻觉重试提示: 如果上一轮生成被检测出幻觉，提供"避免错误"的提示
    3. 异常处理更完善: 区分敏感内容错误和一般错误
    4. 输入来源多样: documents 可能来自向量库检索或网络搜索
"""

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from rag.llm_models.embeddings_model import llm
from rag.utils.logger import log


def generate(state):
    """
    生成回答
    ========
    基于文档和上下文，让 LLM 生成最终的回答文本。

    参数:
        state (GraphState):
            当前图状态，涉及到多个字段:
            - question (str): 用户问题
            - documents (List[Document]): 检索到的相关文档
            - generation (str): 上一轮生成的回答
                * 首次生成时为空字符串 ""
                * 重新生成时包含被判定为幻觉的错误回答
            - messages (List[BaseMessage]): 对话历史
                * 用于构建聊天上下文

    返回:
        dict: 更新后的状态:
            {
                "documents": documents,         # 保持原样
                "question": question,           # 保持原样
                "generation": generated_text,   # LLM 生成的回答
                "messages": [AIMessage(...)],    # 始终写入（包括重试）
            }

    代码逻辑（8 步）:
        1. 提取状态字段
        2. 构建失败提示（如有上一轮幻觉回答）
        3. 构建对话历史摘要（最近 6 条消息）
        4. 格式化文档为字符串
        5. 构建 RAG 提示词
        6. 调用 LLM 生成回答
        7. 异常处理
        8. 返回更新状态
    """
    log.info("--GENERATE---")

    # ===== 1. 提取状态信息 =====
    question = state["question"]                    # 当前用户问题
    documents = state["documents"]                  # 检索到的文档列表
    previous_generation = state.get("generation", "") # 之前生成的回答（首次为空）
    messages = state.get("messages", [])            # 对话历史

    # ===== 2. 构建重试提示 =====
    # 如果存在上一轮被判定为幻觉的回答，提示 LLM 避免相同的错误
    # 这给了 LLM "纠错参考"，让重试更有针对性
    if previous_generation:
        hint = (
            f"\n\n注意：以下回答被判定为存在幻觉（不基于事实），"
            f"请重新回答，避免相同的错误：\n{previous_generation}"
        )
    else:
        hint = ""  # 首次生成，无需提示

    # ===== 3. 构建对话历史摘要 =====
    # 多轮对话时，LLM 需要知道之前的对话内容才能给出连贯的回答。
    # 取最近 6 条消息（约 3 轮对话）作为上下文窗口，平衡完整性和 prompt 长度。
    chat_history = ""
    if messages:
        recent_messages = messages[-6:]
        history_parts = []
        for msg in recent_messages:
            # 只取 human 和 ai 类型的消息，跳过 tool/system 等
            if msg.type == "human":
                role = "用户"
            elif msg.type == "ai":
                role = "助手"
            else:
                continue  # 跳过 tool/system 等不需要在历史中展示的消息

            # AIMessage 如果带 tool_calls，content 可能为空，跳过无内容消息
            content = msg.content if isinstance(msg.content, str) else ""
            if not content:
                continue

            history_parts.append(f"{role}: {content}")

        if history_parts:
            chat_history = (
                "\n\n以下是之前的对话历史，供你参考上下文：\n"
                + "\n".join(history_parts)
                + "\n\n"
            )

    # ===== 4. 构建提示词模板 =====
    prompt = PromptTemplate(
        template=(
            "你是一个问答任务助手，请根据以下检索到的上下文内容回答问题。"
            "如果不知道答案，请直接说明。回答保持简洁"
            "{chat_history}"          # 对话历史（可能为空）
            "问题:{question}\n"
            "上下文:{context}"
            "{hint}"                  # 重试提示（可能为空）
        ),
        # input_variables: 提示词中需要替换的占位符
        # 所有变量必须在 invoke() 时提供
        input_variables=["question", "context", "hint", "chat_history"],
    )

    # ===== 5. 文档格式化函数 =====
    def format_docs(docs):
        """
        将 Document 列表合并为单个字符串，用于填入提示词的 {context} 占位符。

        参数:
            docs: Document 对象或 Document 列表

        返回:
            str: 合并后的文档文本，每个文档用 \n\n 分隔

        处理逻辑:
            - 列表类型: 取每个 doc 的 page_content 属性，用 \n\n 连接
            - 单个 Document: 取 page_content 并用 \n\n 包裹
            - 为什么用 \n\n？两个换行创建视觉分隔，帮助 LLM 区分不同的文档
        """
        if isinstance(docs, list):
            return "\n\n".join(doc.page_content for doc in docs)
        else:
            return "\n\n" + docs.page_content + "\n\n"

    # ===== 6. 构建 RAG 处理链 =====
    # prompt | llm | StrOutputParser(): 标准三步管道
    rag_chain = prompt | llm | StrOutputParser()

    # ===== 7. 执行生成 =====
    try:
        generation = rag_chain.invoke(
            {
                "context": format_docs(documents),
                "question": question,
                "hint": hint,
                "chat_history": chat_history,
            }
        )
    except Exception as e:
        # ===== 8. 异常处理 =====
        error_msg = str(e)

        # 判断是否为内容安全审查错误
        # GemAI/DeepSeek API 遇到敏感内容时返回的错误通常包含 "sensitive" 关键词
        if "sensitive" in error_msg.lower():
            # 敏感内容: 给用户友好的提示
            generation = (
                "抱歉，搜索结果中包含敏感内容，无法生成回答。"
                "请换个问题试试。"
            )
            log.warning("--生成失败：内容包含敏感词--")
        else:
            # 一般错误: 返回错误诊断信息（截断 200 字符）
            generation = f"生成回答时出错：{error_msg[:200]}"
            log.error(f"--生成失败：{error_msg[:200]}--")

    # ===== 9. 构建返回结果 =====
    # 始终写入 messages，即使是幻觉重试。
    # 为什么不跳过重试时的 messages？
    #   如果跳过，修正后的回答永远不进 messages，
    #   后续多轮对话的 chat_history 会读到之前失败的幻觉版本。
    #   两个 AIMessage 都在 messages 中总比只有错误的那个好。
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "messages": [AIMessage(content=generation)],
    }
