"""
Graph2 — Corrective RAG (CRAG) 工作流编排
==========================================
这是第二个 RAG 管道的主流程文件，实现了完整的自校正检索增强生成。

与 Graph1 的核心区别:
    Graph1: 线性流程 + 简单重试（不相关 → 重写 → 重新来）
    Graph2: 多路并行 + 多层评估 + 自校正循环

Graph2 完整流程:
    START
      │
      ▼
    ┌────────────────┐
    │ route_question  │  ① 问题路由：判断走向量库还是网络搜索
    └───┬────────┬───┘
        │        │
  vectorstore  web_search
        │        │
        ▼        │
   ┌──────────┐  │
   │ retrieve  │  │  ② 向量检索 / 网络搜索
   └────┬─────┘  │
        │        │
        ▼        │
   ┌────────────┐ │
   │   grade    │ │  ③ 文档逐条评估，过滤不相关文档
   │ documents  │ │
   └──┬────┬────┘ │
      │    │      │
      │    └── 无相关文档, 次数<2 ──▶ transformer_query → 回到 retrieve
      │    │
      │    └── 无相关文档, 次数≥2 ──▶ web_search ─────────────┐
      │                                                        │
      │ 有相关文档                                              │
      ▼                                                        ▼
   ┌──────────┐◀───────────────────────────────────────────────┘
   │ generate  │  ④ 答案生成
   └────┬─────┘
        │
        ▼
   ┌────────────────────────┐
   │ grade_generation       │  ⑤ 质量评估
   │ (幻觉检测 + 回答评估)    │
   └──┬────────┬───────────┘
      │        │            │
   useful  not_useful  not_supported
      │        │            │
      ▼        ▼            ▼
     END  transformer   generate
          _query          (重试)
          → retrieve

循环控制机制:
    1. transformer_query 循环: 最多 2 次，超过后降级为网络搜索
    2. generate 重试: 幻觉不通过时重新生成（无次数限制，但 LLM 通常 1-2 次即可）
    3. 死循环防护: transforme_count 计数器 + 降级机制确保流程最终能终止

什么是 Corrective RAG？
    Corrective RAG 在基础 RAG 上增加了:
    - 问题路由 (Route): 决定用哪个检索源
    - 文档评分 (Grade): 过滤不相关文档
    - 幻觉检测 (Hallucination Check): 确保回答基于文档
    - 答案评估 (Answer Check): 确保回答解决问题
    - 自动修正 (Corrective Loop): 发现问题后自动调整策略
"""

import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from rag.graph2.generate_node import generate
from rag.graph2.grade_answer_chain import answer_grader_chain
from rag.graph2.grade_documents_node import grade_documents
from rag.graph2.grade_hallucinations_chain import hallucination_grader_chain
from rag.graph2.graph_state2 import GraphState
from rag.graph2.query_rote_chain import question_route_chain
from rag.graph2.retriever_node import retrieve
from rag.graph2.transformer_query_node import transformer_query
from rag.graph2.web_search_node import web_search
from rag.utils._print_event import print_event_graph2
from rag.utils.logger import log


# ==================== 条件路由函数 ====================

def grade_generation_v_documents_and_question(state):
    """
    生成内容质量评估（双阶段评估）
    ==============================
    这是 Graph2 最核心的质量保障函数，分两个阶段评估生成内容。

    阶段1 — 幻觉检测:
        检查生成内容是否基于检索文档（事实集）。
        - 通过: 进入阶段2
        - 不通过: 返回 "not supported" → 触发重新生成

    阶段2 — 回答质量评估:
        检查生成内容是否准确回答了用户问题。
        - 通过: 返回 "useful" → 流程成功结束
        - 不通过: 返回 "not useful" → 触发问题优化，重新检索

    参数:
        state (GraphState):
            当前图状态，需要以下字段:
            - question (str): 用户问题
            - documents (List[Document]): 检索到的参考文档（事实集）
            - generation (str): LLM 生成的回答

    返回:
        Literal["useful", "not useful", "not supported"]:
            三个可能的路由目标:
            - "useful": 回答质量好 → 结束 (END)
            - "not useful": 回答没解决问题 → 优化问题后重新检索
            - "not supported": 回答存在幻觉 → 重新生成

    异常处理策略:
        - 幻觉检测异常 → 默认通过（grade="yes"），因为"漏检"比"误停"危害小
        - 答案评估异常 → 默认通过（grade="yes"），同样原因
    """
    log.info("---检查生成的内容是否存在幻觉---")

    # ===== 阶段1: 幻觉检测 =====
    question = state["question"]     # 用户原始问题
    documents = state["documents"]   # 参考文档（事实集）
    generation = state["generation"] # LLM 生成的回答

    try:
        # 调用幻觉检测链
        # 输入: 事实集(documents) + 生成内容(generation)
        # 输出: GradeHallucinations 对象，包含 binary_score 字段
        score = hallucination_grader_chain.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score.binary_score  # "yes" = 无幻觉, "no" = 有幻觉
    except Exception as e:
        # 检测异常 → 默认通过（宽容策略）
        # 为什么默认通过而不是拒绝？
        #   幻觉检测是辅助性的优化步骤，不是关键路径。
        #   如果检测失败，宁可放过可能有幻觉的回答（用户可以自行判断），
        #   也不要中断流程导致用户得不到任何回答。
        log.warning(f"--幻觉检测异常，默认通过: {str(e)[:100]}--")
        grade = "yes"

    # 分支判断:
    if grade == "yes":
        # 条件成立: 回答基于事实（无幻觉）
        log.info("--判定:生成内容基于参考文档")

        # ===== 阶段2: 回答质量评估 =====
        log.info("--评估:生成回答与内容的匹配度--")
        try:
            # 调用回答评估链
            # 输入: 用户问题 + 生成回答
            # 输出: GradeAnswer 对象
            score = answer_grader_chain.invoke(
                {"question": question, "generation": generation}
            )
            grade = score.binary_score  # "yes" = 解决问题, "no" = 未解决
        except Exception as e:
            log.warning(f"--答案评估异常，默认有用: {str(e)[:100]}--")
            grade = "yes"

        # 分支判断:
        if grade == "yes":
            # 条件成立: 回答解决了用户问题 → 成功！
            log.info("--正确回答问题--")
            return "useful"  # → END（流程结束）
        else:
            # 条件不成立: 回答没解决问题
            # 可能原因: 问题理解有误、文档不够全面、问题需要更精确的表述
            log.info("--判定，生成内容未能准确回答问题--")
            return "not useful"  # → transformer_query（优化问题后重新检索）

    else:
        # 条件不成立: 回答存在幻觉（不基于事实）
        # 可能原因: LLM 编造了文档中没有的内容、文档信息不足时自行脑补
        log.info("--判定:生成内容未基于参考文档，将重新尝试---")
        return "not supported"  # → generate（重新生成回答）


def decide_to_generate(state):
    """
    决定生成还是优化问题（检索后路由）
    =================================
    在 grade_documents 过滤文档后，根据相关文档数量和优化次数决定下一步。

    参数:
        state (GraphState):
            当前状态，关键字段:
            - documents (List[Document]): 过滤后的文档列表
            - transforme_count (int): 已优化问题的次数

    返回:
        Literal["generate", "transformer_query", "web_search"]:
            - "generate": 有相关文档 → 直接生成回答
            - "transformer_query": 无相关文档且次数<2 → 优化问题后重新检索
            - "web_search": 无相关文档且次数≥2 → 降级为网络搜索

    决策树:
        documents 为空?
        ├── 是 → transforme_count >= 2?
        │        ├── 是 → "web_search"   (向量库找不到，用网络搜索)
        │        └── 否 → "transformer_query" (还有优化机会，试试改写问题)
        └── 否 → "generate"  (有相关文档，可以生成答案)

    设计意图:
        这个逐级降级策略确保了流程的"韧性":
        1. 先尝试向量库检索（最快、最可控）
        2. 检索效果不好 → 优化问题再试（给向量库第二次机会）
        3. 还是不行 → 降级为网络搜索（牺牲速度和内容控制，换取信息覆盖）
        4. 最多重写 2 次，防止无限循环
    """
    log.info("决定是否生成还是优化问题")

    # 获取过滤后的文档列表
    filtered_documents = state["documents"]

    # 获取已转换问题的次数（默认为 0）
    transform_count = state.get("transforme_count", 0)

    # 分支判断:
    if not filtered_documents:
        # 条件成立: 没有相关文档（所有文档都被 grade_documents 过滤掉了）

        if transform_count >= 2:
            # 条件成立: 已经优化了 2 次问题还是找不到相关文档
            # → 向量库可能不包含这个领域的知识
            # → 降级为网络搜索，获取更广泛的信息
            log.info("--决策，所有文档都与文档无关，并且已经循环了2次，转为web查询问题")
            return "web_search"
        else:
            # 条件不成立: 还有优化空间（transform_count < 2）
            # → 给向量库另一次机会，用优化后的问题重新检索
            log.info("--决策，所有文档都与问题无关，将转换查询问题--")
            return "transformer_query"
    else:
        # 条件不成立: 有相关文档
        # → 可以基于这些文档生成回答
        log.info("--决策，生成最终回答--")
        return "generate"


def route_question(state):
    """
    问题路由函数（入口路由）
    ========================
    在流程开始时，判断用户问题应该走向量知识库检索还是网络搜索。

    参数:
        state (GraphState):
            当前状态，关键字段:
            - question (str): 用户问题

    返回:
        Literal["vectorstore", "web_search"]:
            - "vectorstore": 走向量知识库检索路径 (retrieve → grade → generate)
            - "web_search": 走网络搜索路径 (web_search → generate)

    路由逻辑:
        LLM 会根据知识库的覆盖范围做判断:
        - AI、机器学习、深度学习、数学基础 → vectorstore
        - 其他领域知识、实时信息 → web_search

    异常处理:
        如果路由判断过程出错 → 默认走向量库检索
        （宁可慢一点查向量库，也不要因路由错误而漏掉可能有用的信息）
    """
    log.info("--- route_question ---")

    question = state["question"]

    try:
        # 调用问题路由链
        # 返回 RouteQuery 对象，包含 datasource 字段
        source = question_route_chain.invoke({"question": question})
    except Exception as e:
        # 路由异常 → 默认走向量检索
        # 为什么默认 vectorstore？
        #   向量库是最可控的检索源，即使领域不完全匹配，
        #   也可能找到部分相关内容。web_search 作为备选更激进。
        log.warning(f"--路由异常，默认走向量检索: {str(e)[:100]}--")
        return "vectorstore"

    # 分支判断:
    if source.datasource == "web_search":
        # 条件成立: LLM 判断需要网络搜索
        # 场景: 问题不在向量库知识范围内（如天气、时事、股票等）
        log.info("--路由到web搜索--")
        return "web_search"
    else:
        # 条件不成立: 包括 "vectorstore" 和任何意外值（防止 LLM 返回预期外的值导致路由失败）
        # 默认走向量检索，安全降级策略
        log.info("--路由到RAG系统--")
        return "vectorstore"


# ==================== 构建工作流图 ====================

# 创建状态图，指定 GraphState 作为状态类型
workflow = StateGraph(GraphState)

# ===== 添加节点 =====
# 6 个节点，每个都有明确的职责:

workflow.add_node("web_search", web_search)
# web_search: 通过网络搜索获取信息
# 来源: rag/graph2/web_search_node.py → web_search 函数
# 触发: route_question 返回 "web_search"，或 decide_to_generate 返回 "web_search"

workflow.add_node("retrieve", retrieve)
# retrieve: 从 Milvus 向量库检索文档
# 来源: rag/graph2/retriever_node.py → retrieve 函数

workflow.add_node("grade_documents", grade_documents)
# grade_documents: 逐文档评估相关性，过滤不相关文档
# 来源: rag/graph2/grade_documents_node.py → grade_documents 函数

workflow.add_node("generate", generate)
# generate: 基于文档生成最终回答
# 来源: rag/graph2/generate_node.py → generate 函数

workflow.add_node("transformer_query", transformer_query)
# transformer_query: 优化用户问题，生成更适合检索的查询
# 来源: rag/graph2/transformer_query_node.py → transformer_query 函数

# ===== 添加边 =====
# 每条边代表一个状态转移

# 边 1: START → 路由判断（条件边）
# 流程一启动就判断走向量库还是网络搜索
workflow.add_conditional_edges(
    START,
    route_question,  # 路由函数，返回 "web_search" 或 "vectorstore" 自行觉得的路由逻辑
    {
        "web_search": "web_search",    # 路由到网络搜索
        "vectorstore": "retrieve",     # 路由到向量检索
    },
)

# 边 2: web_search → generate（固定边）
# 网络搜索完成后直接生成回答（跳过文档评估，因为网络搜索结果格式不同）
workflow.add_edge("web_search", "generate")

# 边 3: retrieve → grade_documents（固定边）
# 检索后立即评估文档相关性
workflow.add_edge("retrieve", "grade_documents")

# 边 4: grade_documents → 决策（条件边）
# 根据文档评估结果决定下一步
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    # 注意: 这里没有映射字典！
    # decide_to_generate 返回 "generate" | "transformer_query" | "web_search"
    # LangGraph 自动将返回值作为目标节点名
)

# 边 5: generate → 质量评估（条件边）
# 生成回答后立即评估质量（幻觉检测 + 回答评估）
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,  # 双阶段评估函数
    {
        "not supported": "generate",           # 有幻觉 → 重新生成
        "useful": END,                         # 质量好 → 结束
        "not useful": "transformer_query",     # 没解决问题 → 优化问题
    },
)

# 边 6: transformer_query → retrieve（固定边）
# 优化问题后重新检索，形成优化循环
# 循环链条: retrieve → grade → [不相关] → transformer_query → retrieve
workflow.add_edge("transformer_query", "retrieve")


# ==================== 记忆功能与编译 ====================

# MemorySaver: 内存中的检查点存储
# 可选替代（生产环境推荐）:
#   - SqliteSaver: 持久化到 SQLite，重启不丢失
#   - RedisSaver: 高性能 + 分布式支持
memory = MemorySaver()

# 会话配置
config = {
    "configurable": {
        "thread_id": str(uuid.uuid4())  # 全局唯一会话 ID
    }
}

# compile: 编译图
# checkpointer=memory: 启用自动状态持久化
graph = workflow.compile(checkpointer=memory)


# ==================== 交互式测试入口 ====================

if __name__ == "__main__":
    """
    命令行交互式测试
    ================
    运行方式: python -m rag.graph2.graph_2

    这个入口支持:
    - 输入问题进行问答
    - 查看完整的流程执行过程（stream_mode="values"）
    - 输入 q / exit / quit 退出

    注意:
    - _printed 集合在每轮对话重新初始化，确保每轮的事件都能打印
    - stream_mode="values" 返回完整状态，适合调试
    - 如果只想要增量变更，可以使用 stream_mode="updates"
    """
    while True:
        question = input("用户:")
        if question.lower() in ["q", "exit", "quit"]:
            print("对话结束，拜拜")
            break
        else:
            _printed = set()  # 每轮重新初始化去重集合

            # 构造 Graph2 的输入状态（5 个字段）
            inputs = {
                "question": question,
                "documents": [],           # 初始空列表
                "generation": "",          # 初始空字符串
                "transforme_count": 0,     # 优化计数器归零
                "messages": [HumanMessage(content=question)],  # 初始消息
            }

            # 执行工作流，stream_mode="values" 返回每步完整状态
            events = graph.stream(inputs, config, stream_mode="values")

            # 打印每个事件
            for event in events:
                print_event_graph2(event, _printed)
