# RAG 智能问答系统 — 项目全面解析文档

---

## 目录

1. [项目概述](#1-项目概述)
2. [项目目录结构](#2-项目目录结构)
3. [技术栈](#3-技术栈)
4. [系统架构图](#4-系统架构图)
5. [核心数据流](#5-核心数据流)
6. [模块逐一详解](#6-模块逐一详解)
   - [6.1 入口与服务层](#61-入口与服务层)
   - [6.2 Graph1 — 基础 RAG 工作流](#62-graph1--基础-rag-工作流)
   - [6.3 Graph2 — Corrective RAG 工作流](#63-graph2--corrective-rag-工作流)
   - [6.4 Agent 定义](#64-agent-定义)
   - [6.5 检索工具](#65-检索工具)
   - [6.6 文档处理与向量库](#66-文档处理与向量库)
   - [6.7 LLM / Embedding 模型](#67-llm--embedding-模型)
   - [6.8 工具类](#68-工具类)
   - [6.9 前端 Web](#69-前端-web)
7. [Graph1 vs Graph2 对比](#7-graph1-vs-graph2-对比)
8. [代码关联关系总结](#8-代码关联关系总结)

---

## 1. 项目概述

这是一个基于 **LangGraph** 构建的 **RAG (检索增强生成) 智能问答系统**，提供了两条不同的检索增强生成管道（Graph1 和 Graph2），通过 FastAPI 提供 SSE 流式接口，前端使用 Vue 3 构建。

**核心能力：**
- 向量知识库检索（Milvus）
- 网络搜索兜底（百度 AI 搜索）
- 问题自动重写优化
- 检索文档相关性评估
- 生成内容幻觉检测
- 回答质量评估与自动修正循环
- 多轮对话记忆

---

## 2. 项目目录结构

```
pythonProject/
├── run_server.py                  # 启动入口
├── .env                           # 环境变量（API Key、模型配置）
├── server/                        # 🌐 FastAPI 后端
│   ├── main.py                    #   FastAPI 应用工厂
│   ├── api.py                     #   REST API 路由
│   ├── schemas.py                 #   Pydantic 请求模型
│   └── stream.py                  #   SSE 流式推送逻辑
├── rag/                           # 🧠 核心 RAG 引擎
│   ├── agent/
│   │   └── rag_agent.py           #   Agent + 工具 + 记忆配置
│   ├── graph/                     #   Graph1: 基础 RAG
│   │   ├── graph1.py              #     主流程编排（状态图）
│   │   ├── graph_state.py         #     状态数据结构定义
│   │   ├── agent_node.py          #     Agent 决策节点
│   │   ├── generate_node.py       #     答案生成节点
│   │   └── rewrite_node.py        #     问题重写节点
│   ├── graph2/                    #   Graph2: Corrective RAG
│   │   ├── graph_2.py             #     主流程编排（状态图）
│   │   ├── graph_state2.py        #     状态数据结构定义
│   │   ├── query_rote_chain.py    #     问题路由链（向量库 vs 网络搜索）
│   │   ├── retriever_node.py      #     文档检索节点
│   │   ├── grade_chain.py         #     文档相关性评分链
│   │   ├── grade_documents_node.py #    文档过滤节点
│   │   ├── transformer_query_node.py # 问题优化节点
│   │   ├── web_search_node.py     #     网络搜索节点
│   │   ├── generate_node.py       #     答案生成节点
│   │   ├── grade_hallucinations_chain.py # 幻觉检测链
│   │   └── grade_answer_chain.py  #     回答质量评估链
│   ├── tools/                     # 🔧 工具
│   │   ├── retriever.py           #     向量检索器（Milvus）
│   │   └── baidu_search_tool.py   #     百度 AI 搜索工具
│   ├── documents/                 # 📄 文档处理
│   │   ├── markdown_parser.py     #     Markdown 解析 + 语义切分
│   │   ├── milvus_db.py           #     Milvus 向量库连接与存储
│   │   └── write_milvus.py        #     多进程批量写入
│   ├── llm_models/
│   │   └── embeddings_model.py    #     LLM + Embedding 模型配置
│   └── utils/                     # 🛠 工具类
│       ├── logger.py              #     日志系统（支持多进程）
│       ├── _print_event.py        #     事件打印/调试输出
│       └── env_utils.py           #     环境变量（Milvus URI）
├── web/                           # 🖥 Vue 3 前端
│   ├── src/
│   │   ├── main.js                #     Vue 入口
│   │   ├── App.vue                #     根组件
│   │   └── components/
│   │       ├── ChatSidebar.vue    #     侧边栏（图选择 + 会话管理）
│   │       ├── ChatMain.vue       #     主聊天区
│   │       └── NodeTimeline.vue   #     节点执行时间线
│   └── vite.config.js             #     Vite 构建配置
└── test/                          # 🧪 测试 Markdown 文档
    ├── test.md ~ test16.md
    └── deep_learning_guide.md
```

---

## 3. 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **LLM** | DeepSeek-Chat (via GemAI API) | 主要对话生成模型 |
| **Embedding** | BAAI/bge-small-zh-v1.5 | 中文语义向量化模型 |
| **向量库** | Milvus + PyMilvus + LangChain-Milvus | 混合检索（密集向量 + BM25 稀疏向量） |
| **图编排** | LangGraph | 状态机驱动的工作流编排 |
| **Agent** | LangChain Tool Calling Agent | 工具调用决策 |
| **后端** | FastAPI + SSE (sse-starlette) | 流式 HTTP 推送 |
| **前端** | Vue 3 + Vite | SPA 聊天界面 |
| **文档解析** | Unstructured + SemanticChunker | Markdown 加载 + 语义切割 |
| **搜索** | 百度 AI 搜索 API | 网络搜索兜底 |
| **日志** | Python logging + RotatingFileHandler | 多进程日志 |

---

## 4. 系统架构图

### Graph1 流程图 (基础 RAG)

```
用户提问
   │
   ▼
┌──────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
│Agent │───▶│ Retrieve  │───▶│  Grade    │───▶│ Generate │───▶│   END    │
│ 判断  │    │  文档检索  │    │ Documents │    │  生成回答  │    │          │
└──────┘    └──────────┘    │  文档评估  │    └──────────┘    └──────────┘
     │                      └─────┬─────┘
     │ 不需要检索                   │ 不相关
     │                            ▼
     │                      ┌──────────┐
     └─────────────────────▶│ Rewrite  │───▶ 回到 Agent
       直接结束               │  问题重写  │
                            └──────────┘
```

### Graph2 流程图 (Corrective RAG)

```
用户提问
   │
   ▼
┌────────────┐
│  路由问题    │
│Route Question│
└──┬──────┬───┘
   │      │
   │      └──────────────┐
   ▼ vectorstore         ▼ web_search
┌──────────┐         ┌───────────┐
│ Retrieve  │         │Web Search │────────┐
│  文档检索  │         │  网络搜索  │         │
└────┬─────┘         └───────────┘         │
     │                                     │
     ▼                                     │
┌────────────┐                             │
│   Grade    │                             │
│ Documents  │                             │
│  文档评估  │                             │
└──┬────┬────┘                             │
   │    │                                  │
   │    └── 无相关文档, 次数<2              │
   │         ┌──────────────┐              │
   │         │Transformer   │──▶ Retrieve  │
   │         │  问题优化     │              │
   │         └──────────────┘              │
   │                                       │
   │    无相关文档, 次数≥2 ──▶ Web Search ──┘
   │
   │ 有相关文档
   ▼
┌──────────┐
│ Generate  │◀──────────────────────────────┘
│  生成回答  │
└────┬─────┘
     │
     ▼
┌────────────────────┐
│  Grade Generation  │
│  质量评估 (幻觉检测) │
└──┬──────┬──────────┘
   │      │            │
   │      │ not useful │ not supported
   │      ▼            ▼
   │  ┌──────────────┐  ┌──────────┐
   │  │Transformer   │  │ Generate │ (重试)
   │  │  问题优化     │  │  重新生成  │
   │  └──────┬───────┘  └──────────┘
   │         │
   │         └──▶ Retrieve
   │
   ▼ useful
┌──────┐
│ END  │
└──────┘
```

### 系统整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        Frontend (Vue 3)                          │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────────────┐    │
│  │ ChatSidebar │   │  ChatMain    │   │  NodeTimeline      │    │
│  │ 图选择+会话  │   │  SSE 消息面板 │   │  节点执行步骤可视化  │    │
│  └─────────────┘   └──────┬───────┘   └────────────────────┘    │
└────────────────────────────┼──────────────────────────────────────┘
                             │ SSE (EventSource)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FastAPI Server (:8000)                        │
│  POST /api/chat  — SSE 流式接口                                  │
│  GET  /api/graphs — 获取可用图列表                                │
│  GET  /health    — 健康检查                                       │
└────────────────────────────┬─────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
┌──────────────────────┐       ┌────────────────────────┐
│      Graph1          │       │        Graph2          │
│  基础 RAG + Agent     │       │  Corrective RAG        │
│  (rag/graph/)        │       │  (rag/graph2/)         │
└──────────┬───────────┘       └───────────┬────────────┘
           │                               │
           └───────────┬───────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    共享组件层                                      │
│  ┌──────────┐ ┌───────────────┐ ┌────────────┐ ┌─────────────┐  │
│  │ Retriever│ │Baike Search   │ │Milvus DB   │ │Markdown     │  │
│  │ 向量检索  │ │  网络搜索工具   │ │  向量存储   │ │  Parser     │  │
│  └──────────┘ └───────────────┘ └────────────┘ └─────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. 核心数据流

一次完整的用户问答过程如下：

```
[前端] → POST /api/chat {question, graph_type, thread_id}
    → [server/stream.py] 调用对应 graph 的 stream() 方法
        → [graph] 以 stream_mode="updates" 迭代执行各节点
            → 每个节点产生 {node_name: node_output} 增量事件
        → 通过 SSE 实时推送给前端
    → [前端] 收到 "node" 事件 → NodeTimeline 组件渲染步骤
    → [前端] 收到 "answer" 事件 → ChatMain 组件渲染回答
    → [前端] 收到 "done" 事件 → 完成
```

---

## 6. 模块逐一详解

### 6.1 入口与服务层

---

#### `run_server.py` — 启动入口

**功能：** 使用 Uvicorn 启动 FastAPI 应用。

**代码逻辑：**
1. 调用 `uvicorn.run()` 启动服务
2. 监听 `0.0.0.0:8000`
3. `reload=True` 开启热重载（开发模式）

**关联：** 入口直接指向 `server.main:app`

---

#### `server/main.py` — FastAPI 应用工厂

**方法：**

| 方法 | 功能 |
|------|------|
| `health()` | `GET /health` — 返回 `{"status": "ok"}` |

**代码逻辑：**
1. 创建 `FastAPI()` 实例，设置标题为 "RAG Chat API"
2. 添加 CORS 中间件，允许所有来源的跨域请求
3. 注册 `server.api` 路由，前缀 `/api`
4. 定义健康检查端点

**关联：** 被 `run_server.py` 引用；引入 `server.api` 路由

---

#### `server/schemas.py` — 请求数据模型

**类：**

| 类名 | 字段 | 说明 |
|------|------|------|
| `ChatRequest` | `question: str` | 用户问题 |
| | `graph_type: str = "graph2"` | 选择工作流，`graph1` 或 `graph2` |
| | `thread_id: Optional[str] = None` | 会话 ID，为 None 时自动生成 |

---

#### `server/api.py` — API 路由

**方法：**

| 方法 | 路径 | 功能 |
|------|------|------|
| `chat()` | `POST /api/chat` | SSE 流式聊天 |
| `list_graphs()` | `GET /api/graphs` | 返回可用图列表 |

**`chat()` 代码逻辑（4 步）：**
1. 接收 `ChatRequest`，读取 `graph_type`
2. 若 `graph_type == "graph1"` → 调用 `stream_graph1(req)`
3. 否则 → 调用 `stream_graph2(req)`
4. 将流式生成器包装成 `EventSourceResponse` 返回

**`list_graphs()` 代码逻辑：**
- 返回两个图的 id、name、description 信息

**关联：** 依赖于 `server.stream` 中的 `stream_graph1()` 和 `stream_graph2()`

---

#### `server/stream.py` — SSE 流式推送

**方法：**

| 方法 | 功能 |
|------|------|
| `_make_event(event_type, data)` | 构造 JSON 格式的 SSE 事件字符串 |
| `_extract_node_detail(node_name, node_output)` | 从节点输出提取可展示的摘要信息 |
| `stream_graph1(req)` | 执行 Graph1 并以 SSE 推送每一步 |
| `stream_graph2(req)` | 执行 Graph2 并以 SSE 推送每一步 |

**`stream_graph1()` 代码逻辑（13 步）：**

```
1. 动态导入 graph1 编译后的 graph 对象
2. 生成/使用 thread_id，构造 LangGraph 的 config
3. yield "session" 事件 (告知前端 thread_id)
4. yield "node" 事件 (标记用户输入)
5. 构造 inputs = {"messages": [("user", question)]}
6. 调用 graph.stream(inputs, config, stream_mode="updates")
7. 遍历 events:
   ├── 每个 event 是 {"node_name": {field: value, ...}}
   ├── 提取 node_name → NODE_LABELS 映射 → 中文标签
   ├── 提取 node_output 摘要信息
   ├── yield "node" 事件
   └── 若 node_name == "generate":
       ├── 提取 generation 内容
       └── yield "answer" 事件
8. 若未拿到回答 → 从 graph.get_state(config) 获取最终状态
9. 回溯 messages 列表找到最后的 AI 消息
10. 若仍无回答 → yield 错误提示
11. 异常处理 → yield "error" 事件
12. yield "done" 事件
```

**`stream_graph2()` 代码逻辑（12 步）：**

```
1. 动态导入 graph2 编译后的 graph 对象
2. 生成/使用 thread_id
3. yield "session" 事件
4. yield "node" 事件
5. 构造 inputs (包含 question, documents, generation, transforme_count, messages)
6. 调用 graph.stream(inputs, config, stream_mode="updates")
7. 遍历 events:
   ├── 提取 node_name, label, detail
   ├── yield "node" 事件
   └── 若 node_name == "generate" → yield "answer" 事件
8. 若未拿到回答 → yield 错误提示
9. 异常处理 → yield "error" 事件
10. yield "done" 事件
```

**SSE 事件类型：**

| 事件类型 | 携带数据 | 触发时机 |
|----------|---------|---------|
| `session` | `{thread_id}` | 会话开始 |
| `node` | `{node, label, detail}` | 每个节点执行完毕 |
| `answer` | `{generation}` | 生成完成 |
| `done` | `{}` | 流结束 |
| `error` | `{message}` | 异常 |

**关联：** 被 `server/api.py` 调用；依赖于 `rag.graph.graph1` 和 `rag.graph2.graph_2`

---

### 6.2 Graph1 — 基础 RAG 工作流

#### `rag/graph/graph_state.py` — 状态数据结构

**类：**

| 类名 | 作用 |
|------|------|
| `AgentState(TypedDict)` | 定义图中流转的状态结构 |
| `Grade(BaseModel)` | 文档相关性评分的结构化输出模型 |

**AgentState 结构：**
```python
messages: Annotated[list[BaseMessage], add_messages]
```
- `add_messages` 是 LangGraph 的内置 reducer，表示新消息会"追加"到现有列表，而非覆盖

**Grade 结构：**
```python
binary_score: str  # "yes" 或 "no"
```

**关联：** 被 `graph1.py` 中的 `grade_documents()` 和 LLM 结构化输出使用

---

#### `rag/graph/graph1.py` — Graph1 主流程编排

这是 Graph1 的核心文件，负责定义所有节点、边和工作流的编译。

**方法/函数：**

| 函数 | 功能 |
|------|------|
| `get_last_human_message(messages)` | 获取消息列表中最后一条人类消息 |
| `grade_documents(state)` | 动态路由：评估文档相关性，决定去 generate 还是 rewrite |

**`get_last_human_message()` 代码逻辑（3 步）：**
```
1. 反向遍历 messages 列表
2. 找到第一个 HumanMessage 实例
3. 返回该 HumanMessage；若找不到则抛出 ValueError
```

**`grade_documents()` 代码逻辑（6 步）：**
```
1. 打印日志 "检查document的相关性"
2. 创建带结构化输出的 LLM (with_structured_output(Grade))
3. 构建评分提示词模板 (PromptTemplate)
   - 输入：context(文档内容) 和 question(用户问题)
   - 要求 LLM 输出 yes/no 的二元评分
4. 构建处理链: prompt | llm_with_structured
5. 从 state["messages"] 获取最后一条消息(检索结果)
   和最后一条人类消息(原始问题)
6. 调用 chain.invoke() 获得评分
   - score == "yes" → return "generate"
   - score != "yes" → return "rewrite"
```

**工作流节点定义（6 行）：**
```python
workflow.add_node("agent", agent)           # 来自 agent_node.py
workflow.add_node("retrieve", ToolNode(...)) # LangGraph 内置，包装检索工具
workflow.add_node("rewrite", rewrite)        # 来自 rewrite_node.py
workflow.add_node("generate", generate)      # 来自 generate_node.py
```

**工作流边定义（5 条）：**
```python
START → agent                                    # 固定边
agent → tools_condition → {"tools": retrieve, END: END}  # 条件边
retrieve → grade_documents → generate/rewrite    # 条件边
rewrite → agent                                   # 固定边（回到 agent 重新判断）
generate → END                                    # 固定边
```

**完整流程：**
```
START → agent → (需要检索?) → retrieve → (文档相关?) → generate → END
       ↑         ↓ 不需要                     ↓ 不相关        ↓
       │       [END]                      rewrite ───→ agent
       └──────────────────────────────────────────────┘
```

**Memory 配置：**
- 使用 `MemorySaver()` 实现对话记忆（内存存储，重启消失）
- 通过 `thread_id` 区分不同会话

**关联：** 被 `server/stream.py` 导入；依赖 `agent_node.py`、`generate_node.py`、`rewrite_node.py`、`retriever.py`

---

#### `rag/graph/agent_node.py` — Agent 决策节点

**方法：**

| 方法 | 功能 |
|------|------|
| `agent(state)` | 调用 LLM 决定是否需要检索工具 |

**代码逻辑（5 步）：**
```
1. 打印日志 "调用智能体"
2. 从 state["messages"] 获取完整消息历史
3. 绑定检索工具到 LLM: llm.bind_tools([retriever_tool])
4. 调用 model.invoke(messages) — LLM 根据上下文决定：
   - 需要检索 → 返回 tool_call (调用 retriever_tool)
   - 不需要 → 直接返回 AI 消息
5. 返回 {"messages": [response]}
```

**关联：** 被 `graph1.py` 注册为 "agent" 节点；依赖 `retriever_tool`

---

#### `rag/graph/generate_node.py` — 答案生成节点（Graph1）

**方法：**

| 方法 | 功能 |
|------|------|
| `generate(state)` | 基于检索到的文档生成最终回答 |

**代码逻辑（5 步）：**
```
1. 打印日志 "生成答案"
2. 从 messages 中提取：
   - question: 最后一条 HumanMessage 的内容
   - docs: 最后一条消息的内容（检索结果文档）
3. 构建 RAG 提示词模板：
   "你是一个问答任务助手。请根据以下检索到的上下文内容回答问题。
   如果不知道答案，请直接说明。回答保持简洁
   问题:{question} 上下文:{context}"
4. 构建处理链: prompt | llm | StrOutputParser()
5. 调用 rag_chain.invoke() 生成回答
6. 返回 {"messages": [response]}
```

**关联：** 被 `graph1.py` 注册为 "generate" 节点

---

#### `rag/graph/rewrite_node.py` — 问题重写节点

**方法：**

| 方法 | 功能 |
|------|------|
| `rewrite(state)` | 优化问题表述，生成更好的检索查询 |

**代码逻辑（4 步）：**
```
1. 打印日志 "转换查询"
2. 从 messages 获取最后一条 HumanMessage 内容（原始问题）
3. 构造 HumanMessage 提示词：
   "分析输入并尝试理解潜在的语义意图/含义。
   这是初始问题: {question}
   请提出一个改进后的问题"
4. 调用 llm.invoke(msg) 获得改进后的问题
5. 返回 {"messages": [response]}
```

**关联：** 被 `graph1.py` 注册为 "rewrite" 节点；当文档不相关时触发

---

### 6.3 Graph2 — Corrective RAG 工作流

#### `rag/graph2/graph_state2.py` — 状态数据结构

**类：**

| 类名 | 作用 |
|------|------|
| `GraphState(TypedDict)` | Graph2 的状态结构 |

**GraphState 结构：**
```python
question: str                         # 当前用户问题
transforme_count: int                 # 问题转换次数计数器
generation: str                       # LLM 生成的回答
documents: List[Document]             # 检索到的文档列表
messages: Annotated[list[BaseMessage], add_messages]  # 对话历史
```

**与 Graph1 状态的区别：**
- Graph1 只有 `messages` 字段
- Graph2 新增 `question`、`generation`、`documents`、`transforme_count` 字段
- Graph2 的状态更结构化，方便各节点独立读写

---

#### `rag/graph2/graph_2.py` — Graph2 主流程编排

这是 Graph2 的核心文件，实现了完整的 Corrective RAG (CRAG) 流程。

**方法/函数：**

| 函数 | 功能 |
|------|------|
| `route_question(state)` | 起始路由：判断走向量检索还是网络搜索 |
| `decide_to_generate(state)` | 评估文档后决定：生成、重写问题、还是网络搜索 |
| `grade_generation_v_documents_and_question(state)` | 生成后质量评估：幻觉检测 + 回答质量 |

---

**`route_question()` 代码逻辑（5 步）：**
```
1. 打印日志 "route_question"
2. 从 state["question"] 获取用户问题
3. 调用 question_route_chain.invoke({"question": question})
   → LLM 判断问题属于 AI/ML/数学领域（向量库）还是其他（网络搜索）
4. 根据 source.datasource 返回:
   - "web_search" → 路由到网络搜索节点
   - "vectorstore" → 路由到向量检索节点
5. 异常时默认返回 "vectorstore"
```

**`decide_to_generate()` 代码逻辑（4 步）：**
```
1. 打印日志 "决定是否生成还是优化问题"
2. 获取 filtered_documents（评估后的相关文档列表）
3. 获取 transforme_count（已转换问题的次数）
4. 判断：
   - 无相关文档 且 转换次数 ≥ 2 → return "web_search"
   - 无相关文档 且 转换次数 < 2 → return "transformer_query"
   - 有相关文档 → return "generate"
```

**`grade_generation_v_documents_and_question()` 代码逻辑（5 步）：**
```
阶段1 — 幻觉检测：
  1. 获取 question、documents、generation
  2. 调用 hallucination_grader_chain.invoke()
  3. 若 grade == "yes" (基于文档) → 进入阶段2
     若 grade != "yes" (存在幻觉) → return "not supported" (触发重新生成)

阶段2 — 回答质量评估：
  4. 调用 answer_grader_chain.invoke()
  5. 若 grade == "yes" → return "useful" (成功，结束)
     若 grade != "yes" → return "not useful" (触发问题优化)
```

---

**工作流节点定义（6 个）：**
```python
workflow.add_node("web_search", web_search)          # 网络搜索
workflow.add_node("retrieve", retrieve)               # 向量检索
workflow.add_node("grade_documents", grade_documents)  # 文档评估
workflow.add_node("generate", generate)                # 答案生成
workflow.add_node("transformer_query", transformer_query) # 问题优化
```

**工作流边定义（6 条）：**
```python
START → route_question → web_search / retrieve               # 条件边
web_search → generate                                         # 固定边
retrieve → grade_documents                                    # 固定边
grade_documents → decide_to_generate → generate/transformer_query/web_search  # 条件边
generate → grade_generation... → useful:END / not_useful:transformer_query / not_supported:generate  # 条件边
transformer_query → retrieve                                  # 固定边（形成循环）
```

**核心循环机制：**
```
不相关文档 → 问题优化 → 重新检索 → 文档评估 → ... （最多优化2次）
    └→ 若2次优化后仍无相关文档 → 走网络搜索

生成有幻觉 → 重新生成
生成答案不准确 → 问题优化 → 重新检索 → ... （循环）
```

---

#### `rag/graph2/query_rote_chain.py` — 问题路由链

**类：**

| 类名 | 作用 |
|------|------|
| `RouteQuery(BaseModel)` | 路由决策的结构化输出 |

**核心对象：**
- `structured_llm_router`：绑定了 `RouteQuery` 结构化输出的 LLM
- `question_route_chain`：`route_prompt | structured_llm_router` 管道

**代码逻辑：**
```
1. 定义 RouteQuery 模型：datasource ∈ {"vectorstore", "web_search"}
2. 设置系统提示词：
   - 向量库包含：AI、机器学习、深度学习、数学基础等知识
   - 这些主题 → vectorstore
   - 其他主题 → web_search
3. 构建处理链：prompt | structured_llm_router
4. 导出 question_route_chain 供 graph_2.py 使用
```

---

#### `rag/graph2/retriever_node.py` — 文档检索节点

**方法：**

| 方法 | 功能 |
|------|------|
| `retrieve(state)` | 调用 Milvus 检索器获取相关文档 |

**代码逻辑（3 步）：**
```
1. 打印日志 "检索库检索"
2. 从 state["question"] 获取问题
3. 调用 retriever.invoke(question) → 返回最相似的 3 个文档
4. 返回 {"documents": documents, "question": question}
```

---

#### `rag/graph2/grade_chain.py` — 文档相关性评分链

**类：**

| 类名 | 作用 |
|------|------|
| `GradeDocuments(BaseModel)` | 文档相关性评分模型 |

**核心对象：**
- `structured_llm_grader`：绑定了 `GradeDocuments` 结构化输出的 LLM
- `retrieval_grade_chain`：评分处理链

**代码逻辑：**
```
1. 定义 GradeDocuments 模型：binary_score ∈ {"yes", "no"}
2. 系统提示词要求 LLM 判断文档是否与问题相关
3. 构建处理链：grade_prompt | structured_llm_grader
4. 导出 retrieval_grade_chain 供 grade_documents_node.py 使用
```

---

#### `rag/graph2/grade_documents_node.py` — 文档过滤节点

**方法：**

| 方法 | 功能 |
|------|------|
| `grade_documents(state)` | 遍历文档，逐个评分并过滤不相关文档 |

**代码逻辑（5 步）：**
```
1. 打印日志 "CHECK DOCUMENT RELEVANCE TO QUESTION"
2. 从 state 获取 question 和 documents
3. 遍历 documents:
   ├── 对每个 doc 调用 retrieval_grade_chain.invoke()
   ├── grade == "yes" → 加入 filtered_docs
   ├── grade != "yes" → 丢弃
   └── 异常时默认保留该文档
4. 返回 {"documents": filtered_docs, "question": question}
```

---

#### `rag/graph2/transformer_query_node.py` — 问题优化节点

**方法：**

| 方法 | 功能 |
|------|------|
| `transformer_query(state)` | 将原始问题改写为更适合检索的优化版本 |

**代码逻辑（5 步）：**
```
1. 打印日志 "优化问题"
2. 从 state 获取 question 和 transforme_count
3. 构建问题重写提示词：
   "作为问题重写器，您需要将输入问题转换为更适合向量数据库检索的优化版本
   请分析输入问题并理解其背后的语义意图/真实含义"
4. 构建处理链：re_write_prompt | llm | StrOutputParser()
5. 调用 question_rewriter.invoke() 获取优化后的问题
6. 返回 {"documents": documents, "question": better_question,
          "transforme_count": transforme_count + 1}
```

**与 Graph1 rewrite_node 的区别：**
- Graph1：输出新的 HumanMessage 放入 messages 列表
- Graph2：直接替换 state 中的 question 字段，并维护计数器防止死循环

---

#### `rag/graph2/web_search_node.py` — 网络搜索节点

**方法：**

| 方法 | 功能 |
|------|------|
| `web_search(state)` | 使用百度 AI 搜索进行联网检索 |

**代码逻辑（4 步）：**
```
1. 打印日志 "网络搜索"
2. 从 state["question"] 获取问题
3. 调用 baidu_search_tool.invoke({"query": question})
4. 将搜索结果包装为 Document 对象
5. 返回 {"documents": [web_results], "question": question}
```

---

#### `rag/graph2/generate_node.py` — 答案生成节点（Graph2）

**方法：**

| 方法 | 功能 |
|------|------|
| `generate(state)` | 基于文档、对话历史和失败提示生成回答 |

**代码逻辑（8 步）：**
```
1. 打印日志 "GENERATE"
2. 从 state 获取 question、documents、previous_generation、messages
3. 构建失败提示 hint：
   - 若 previous_generation 存在 → 添加避免幻觉的提示
   - 否则为空
4. 构建对话历史摘要 chat_history：
   - 取最近 6 条消息
   - 格式化为 "用户: ... / 助手: ..."
5. 定义内部函数 format_docs(docs)：
   - 将 Document 列表合并为一个字符串（page_content 用 \n\n 分隔）
6. 构建处理链：prompt | llm | StrOutputParser()
7. 调用 rag_chain.invoke() 生成回答
8. 异常处理：敏感内容 / 其他错误
9. 仅在首次生成时写入 messages（避免幻觉重试时堆积冗余消息）
10. 返回 {"documents", "question", "generation", 可选"messages"}
```

**与 Graph1 generate_node 的区别：**
- Graph1：简单单一，仅从 messages 获取上下文
- Graph2：支持对话历史 (chat_history)、幻觉重试提示 (hint)、敏感词异常处理、条件消息写入

---

#### `rag/graph2/grade_hallucinations_chain.py` — 幻觉检测链

**类：**

| 类名 | 作用 |
|------|------|
| `GradeHallucinations(BaseModel)` | 幻觉检测结果模型 |

**核心对象：**
- `structured_llm_grade`：结构化输出的 LLM
- `hallucination_grader_chain`：检测生成内容是否基于检索文档

**代码逻辑：**
```
1. 定义 GradeHallucinations 模型：binary_score ∈ {"yes", "no"}
2. 系统提示词要求 LLM 判断：
   - "回答是否基于/支持于给定事实集"
3. 输入：documents (事实集) + generation (生成内容)
4. 导出 hallucination_grader_chain 供 graph_2.py 使用
```

---

#### `rag/graph2/grade_answer_chain.py` — 回答质量评估链

**类：**

| 类名 | 作用 |
|------|------|
| `GradeAnswer(BaseModel)` | 回答质量评分模型 |

**核心对象：**
- `structured_llm_grader`：结构化输出的 LLM
- `answer_grader_chain`：评估回答是否解决了用户问题

**代码逻辑：**
```
1. 定义 GradeAnswer 模型：binary_score ∈ {"yes", "no"}
2. 系统提示词要求 LLM 判断：
   - "回答是否解决了该问题"
3. 输入：question (用户问题) + generation (生成回答)
4. 导出 answer_grader_chain 供 graph_2.py 使用
```

---

### 6.4 Agent 定义

#### `rag/agent/rag_agent.py` — RAG Agent

**功能：** 定义带工具调用和会话记忆的 LangChain Agent。

**关键组件：**

| 组件 | 作用 |
|------|------|
| `prompt` | ChatPromptTemplate：系统提示 + 历史记录占位 + 用户输入 + 工具记录占位 |
| `agent` | `create_tool_calling_agent()` 创建的 Agent |
| `executor` | AgentExecutor：执行器，管理工具调用循环 |
| `get_session_history()` | 会话历史获取函数 |
| `agent_with_history` | `RunnableWithMessageHistory`：带记忆的 Agent |

**代码逻辑（测试代码）：**
```
1. 定义 prompt 模板（4 个占位符）：
   - system: "你是一个智能助手..."
   - chat_history: 历史对话
   - human: "{input}"
   - agent_scratchpad: 工具调用记录
2. 创建 tool calling agent: create_tool_calling_agent(llm, [retriever_tool], prompt)
3. 创建 executor: AgentExecutor(agent=agent, tools=[retriever_tool])
4. 定义 get_session_history()：内存字典存储，按 session_id 返回 ChatMessageHistory
5. 包装为 RunnableWithMessageHistory：自动管理会话记忆
```

**注意：** 这个文件在当前工作流中**并不直接被 graph1 使用**。
Graph1 直接在 `agent_node.py` 中实现了 agent 逻辑（`llm.bind_tools` + `model.invoke`）。
此文件更多是独立的 Agent 演示/测试代码。

---

### 6.5 检索工具

#### `rag/tools/retriever.py` — 向量检索器

**功能：** 创建基于 Milvus 的检索工具。

**代码逻辑（4 步）：**
```
1. 创建 MilvusVectorSave 实例 → 连接 Milvus
2. 调用 mv.create_connection() → 初始化 Milvus 连接和 vector_store_saved
3. 从 vector_store_saved 获取 retriever:
   - search_type: "similarity" (向量相似度搜索)
   - k: 3 (返回最相似的 3 个文档)
   - score_threshold: 0.1 (相似度最低阈值)
   - ranker_type: "rrf" (RRF 混合重排序)
   - filter: {"category": "content"} (只检索 content 类别)
4. 使用 create_retriever_tool() 包装为 LangChain Tool
   - 名称: "rag_retriever"
   - 描述: "搜索关于人工智能、机器学习、深度学习、数学基础等相关知识"
```

**关联：** 被 `agent_node.py`、`graph2/retriever_node.py` 和 `rag_agent.py` 引用

---

#### `rag/tools/baidu_search_tool.py` — 百度 AI 搜索工具

**方法：**

| 方法 | 功能 |
|------|------|
| `baidu_search_tool(query)` | 调用百度 AI 搜索 API 进行联网搜索 |

**代码逻辑（5 步）：**
```
1. 使用 @tool 装饰器注册为 LangChain Tool
2. 构造请求 URL: https://qianfan.baidubce.com/v2/ai_search
3. 设置 Authorization Header (Bearer Token)
4. 设置搜索参数：
   - search_source: "baidu_search_v2"
   - search_recency_filter: "month" (只搜索最近一个月)
5. POST 请求 → 成功返回 response.text，失败返回错误信息
```

**关联：** 被 `graph2/web_search_node.py` 使用

---

### 6.6 文档处理与向量库

#### `rag/documents/milvus_db.py` — Milvus 向量数据库

**类：**

| 类名 | 功能 |
|------|------|
| `MilvusVectorSave` | Milvus 连接管理 + 文档存储 |

**方法：**

| 方法 | 功能 |
|------|------|
| `create_collection()` | 自定义 schema 创建 collection |
| `create_connection()` | 创建 LangChain-Milvus 连接 |
| `add_document(datas)` | 批量插入 Document |

---

**`create_collection()` 代码逻辑（8 步）：**
```
1. 连接 Milvus: MilvusClient(uri=MILVUS_URI)
2. 创建 Schema，启用动态字段
3. 添加字段：
   ├── id (INT64, 主键, 自增)
   ├── text (VARCHAR(6000), 启用 jieba 分词 + cnalphanumonly 过滤器)
   ├── category, source, filename, filetype, title (VARCHAR 元数据)
   ├── category_depth (INT64)
   ├── sparse (SPARSE_FLOAT_VECTOR) — BM25 稀疏向量
   └── dense (FLOAT_VECTOR, dim=512) — 密集向量
4. 添加 BM25 Function：text → sparse 自动编码
5. 创建稀疏向量索引 (SPARSE_INVERTED_INDEX, BM25)
6. 创建密集向量索引 (HNSW, IP 内积)
7. 若 collection 已存在 → 先删除再重建
8. 创建 collection
```

**`create_connection()` 代码逻辑（4 步）：**
```
1. 创建 MilvusClient 连接
2. 解析 URI 获取 host 和 port
3. 调用 connections.connect() 建立连接
4. 创建 LangChain Milvus 实例：
   - embedding_function: bge_embedding (BAAI/bge-small-zh-v1.5)
   - vector_field: ["dense", "sparse"] （混合检索）
   - builtin_function: BM25BuiltInFunction()
   - consistency_level: "Strong" （强一致性）
   - auto_id: True, enable_dynamic_field: True
```

**`add_document()` 代码逻辑（3 步）：**
```
1. 遍历 documents
2. 若 metadata 中有 languages 字段 → 转换为逗号分隔字符串
3. 调用 vector_store_saved.add_documents(datas) 批量插入
```

**关联：** 被 `retriever.py`、`write_milvus.py` 引用

---

#### `rag/documents/markdown_parser.py` — Markdown 文档解析器

**类：**

| 类名 | 功能 |
|------|------|
| `MarkdownParser` | Markdown 文件加载、解析、语义切分 |

**方法：**

| 方法 | 功能 |
|------|------|
| `parse_mark_down(md_file)` | 使用 Unstructured 加载 MD 文件 |
| `merge_title_content(datas)` | 将标题与内容合并 |
| `text_chunker(datas)` | 对长文档进行语义切分 |
| `markdown_to_documents(md_file)` | 完整流程：解析 → 合并 → 切分 |

---

**`parse_mark_down()` 代码逻辑（3 步）：**
```
1. 创建 UnstructuredLoader 实例
   - model: "elements" (按文档元素分割)
   - strategy: "fast" (快速模式)
2. 调用 loader.lazy_load() 懒加载
3. 返回 Document 列表
```

**`merge_title_content()` 代码逻辑（6 步）：**
```
1. 初始化 merged_datas 列表和 parent_dict 字典
2. 遍历所有 documents:
   ├── 清理 metadata 中的 languages 字段
   ├── category == "NarrativeText" 且无 parent → 独立内容文档
   ├── category == "Title":
   │   ├── 在 metadata 中标记 title
   │   ├── 若有 parent_id → 拼接 "父标题 -> 子标题"
   │   └── 将当前标题记录到 parent_dict（它可能是后面内容的父级）
   └── category == "Title" 以外 且 有 parent_id:
       ├── 拼接 "父标题内容 + 自身内容"
       ├── 标记父文档的 category 为 "content"（用于后续检索过滤）
       └── 加入 merged_datas
3. 将 parent_dict 中的标题文档也加入结果
4. 返回 merged_datas
```

**`text_chunker()` 代码逻辑（3 步）：**
```
1. 初始化 docs 列表
2. 遍历 documents:
   ├── 若 page_content 长度 > 100：
   │   └── 调用 SemanticChunker.split_documents() 语义切分
   └── 无论是否切割，原文档都加入列表
3. 返回 docs
```

**`markdown_to_documents()` 代码逻辑（3 步）：**
```
parse_mark_down() → merge_title_content() → text_chunker() → 返回最终文档列表
```

**关联：** 被 `milvus_db.py` 和 `write_milvus.py` 使用

---

#### `rag/documents/write_milvus.py` — 多进程批量写入

**方法：**

| 方法 | 功能 |
|------|------|
| `file_parser_process(dir_path, output_queue, batch_size)` | 子进程1：解析目录下所有 MD 文件 |
| `milvus_writer_process(input_queue)` | 子进程2：从队列读取并写入 Milvus |

---

**`file_parser_process()` 代码逻辑（6 步）：**
```
1. 扫描目录下所有 .md 文件
2. 若无 .md 文件 → 发送 None 终止信号
3. 创建 MarkdownParser 实例
4. 遍历 md 文件：
   ├── 调用 parser.markdown_to_documents(md_path) 解析
   ├── 将解析出的 docs 累积到 doc_batch
   └── 当 doc_batch 达到 batch_size (20) → 放入队列，清空缓冲区
5. 将剩余文档也放入队列
6. 发送 None 终止信号
```

**`milvus_writer_process()` 代码逻辑（5 步）：**
```
1. 创建 MilvusVectorSave 实例
2. 调用 mv.create_connection()
3. 循环从队列获取数据：
   ├── 若为 None → break (终止信号)
   └── 调用 mv.add_document(datas) 写入
4. 累计计数并打印进度
5. 打印最终写入总数
```

**主流程（`__main__` 代码逻辑）：**
```
1. 配置目录路径和队列大小
2. 先创建 collection
3. 创建进程间通信 Queue (maxsize=20)
4. 启动两个子进程：
   - parse_process: 解析 MD 文件
   - write_process: 写入 Milvus
5. 等待两个进程结束
```

---

### 6.7 LLM / Embedding 模型

#### `rag/llm_models/embeddings_model.py` — 模型配置

**核心对象：**

| 对象 | 说明 |
|------|------|
| `openai_embedding` | OpenAI 兼容 API 的 Embedding 模型 ([官逆]gpt-4o-mini) |
| `bge_embedding` | BAAI/bge-small-zh-v1.5 — 中文语义向量模型（512维） |
| `llm` | ChatOpenAI 实例，使用 deepseek-chat 模型 |

**关键配置：**
```python
# Embedding
bge_embedding:
  - model: "BAAI/bge-small-zh-v1.5"
  - device: "cpu"
  - normalize_embeddings: True (归一化后余弦相似度 = 内积)

# LLM
llm:
  - model: "deepseek-chat"
  - base_url: "https://api.gemai.cc/v1"
  - temperature: 0.5
```

**关联：** 被所有需要使用 LLM 或 Embedding 的模块导入

---

### 6.8 工具类

#### `rag/utils/env_utils.py` — 环境变量

```python
MILVUS_URI = 'http://localhost:19530'
COLLECTION_NAME = "rag_table"
```

---

#### `rag/utils/logger.py` — 日志系统

**类：**

| 类名 | 功能 |
|------|------|
| `Logger` | 支持多进程的单例日志器 |

**方法：**

| 方法 | 功能 |
|------|------|
| `_add_console_handler()` | 添加控制台输出 |
| `_add_file_handler()` | 添加文件输出（按日轮转） |
| `debug/info/warning/error/critical(msg)` | 各级别日志输出 |
| `exception(msg)` | 异常日志 |
| `get_logger()` | 返回 logging.Logger 实例 |

**便捷函数：**
- `get_logger(name, log_dir, level)` → 工厂函数
- `init_logger(name, log_dir, level, console, file)` → 初始化并返回 logging.Logger
- `log` → 预创建的默认日志实例（`get_logger('default', 'logs', INFO)`）

**特性：**
- 单例模式：同一 name 只创建一个实例
- 控制台 + 文件双输出
- `RotatingFileHandler`：单文件最大 10MB，保留 5 个备份
- 支持多进程
- 日志格式：`时间 - 进程名 - 进程ID - 日志器名 - 级别 - 消息`

---

#### `rag/utils/_print_event.py` — 事件打印工具

**方法：**

| 方法 | 功能 |
|------|------|
| `print_event(event, _printed, max_length)` | 打印 Graph1 事件（格式：messages） |
| `print_event_graph2(event, _printed, max_length)` | 打印 Graph2 事件（格式：question/generation/documents/messages） |

**`print_event()` 代码逻辑：**
```
1. 从 event 获取 "dialog_state" 和 "messages"
2. 若存在 dialog_state → 打印当前对话状态
3. 若存在 messages：
   ├── 取最后一条消息
   ├── 若消息 id 未打印过（去重）：
   │   ├── 调用 message.pretty_repr(html=True) 格式化
   │   ├── 超过 max_length(1500) 则截断
   │   └── 打印并记录到 _printed 集合
```

**`print_event_graph2()` 代码逻辑：**
```
1. 从 event 获取 question, generation, documents, messages
2. question 未打印过 → 打印 "问题: {question}"
3. documents 存在 → 打印 "检索到 N 个文档"
4. generation 未打印过 → 打印 "回答: {generation}" (超长截断)
5. messages 存在 → 同 print_event 逻辑，打印最后一条消息
```

---

### 6.9 前端 Web

#### 文件结构
```
web/
├── index.html
├── vite.config.js
├── dist/                        # 构建产物
└── src/
    ├── main.js                  # Vue 应用入口
    ├── App.vue                  # 根组件
    └── components/
        ├── ChatSidebar.vue      # 侧边栏（图选择 + 会话列表）
        ├── ChatMain.vue         # 主聊天区（SSE 连接 + 消息渲染）
        └── NodeTimeline.vue     # 节点执行时间线（可视化工作流步骤）
```

**功能概述：**
- **ChatSidebar**：选择使用 Graph1 还是 Graph2，管理多个会话
- **ChatMain**：通过 EventSource 连接 SSE 端点，实时渲染回答和节点步骤
- **NodeTimeline**：以时间线形式展示 LangGraph 中各节点的执行顺序和状态

---

## 7. Graph1 vs Graph2 对比

| 特性 | Graph1 (基础 RAG) | Graph2 (Corrective RAG) |
|------|-------------------|-------------------------|
| **启动方式** | Agent 判断是否检索 | 直接路由决策 |
| **检索策略** | 仅向量库 | 向量库 + 网络搜索双路 |
| **文档评估** | 简单一次评估 | 逐文档评估，过滤不相关 |
| **问题优化** | 简单重写，回到 Agent 重新判断 | 结构化重写，维护转换次数上限（2次） |
| **幻觉检测** | ❌ 无 | ✅ 检测生成内容是否基于文档 |
| **回答评估** | ❌ 无 | ✅ 评估回答是否解决用户问题 |
| **纠错机制** | 不相关 → 重写问题 → 重新检索 | 幻觉 → 重新生成；不准确 → 优化问题 → 重新检索 |
| **循环防护** | 无显式防护 | transforme_count 限制（最多优化2次） |
| **状态结构** | 简单（仅 messages 列表） | 结构化（question + documents + generation + count） |
| **适用场景** | 简单问答，知识明确 | 复杂问答，需要多轮修正 |

---

## 8. 代码关联关系总结

```
run_server.py
    │
    ▼
server/main.py  ←── server/api.py  ←── server/schemas.py
                       │
                       ▼
                  server/stream.py
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   rag/graph/graph1.py       rag/graph2/graph_2.py
   (graph 对象)               (graph 对象)
          │                         │
    ┌─────┼─────┐          ┌────────┼────────┐
    ▼     ▼     ▼          ▼        ▼        ▼
  agent_ generate rewrite  retriever grade   generate
  node   _node   _node    _node    _docs    _node
    │     │      │         │        │         │
    └─────┼──────┘         │        │         │
          │                │        │         │
          ▼                │        │         │
   rag/tools/retriever.py  │        │         │
          │                │        │         │
          ▼                │        │         │
   rag/documents/milvus_db.py      │         │
          │                         │         │
          ▼                         ▼         ▼
   rag/llm_models/embeddings_model.py (LLM + Embedding)
          │
          ├── bge_embedding → BAAI/bge-small-zh-v1.5 (向量化)
          └── llm → deepseek-chat via GemAI API (生成/评估)

   共享工具层：
   ├── rag/tools/baidu_search_tool.py ←── graph2/web_search_node.py
   ├── rag/documents/markdown_parser.py ←── milvus_db.py, write_milvus.py
   ├── rag/utils/logger.py ←── 所有模块
   ├── rag/utils/env_utils.py ←── milvus_db.py
   └── rag/utils/_print_event.py ←── graph1.py, graph_2.py (调试)

   数据流方向：
   用户输入 → API → stream → graph.stream() → 各节点迭代
       → (LLM API / Milvus / 百度搜索) → 节点输出 → SSE 推送 → 前端
```

### 依赖链总结

```
前端 Vue 3 (SSE EventSource)
    ↓
FastAPI (/api/chat)
    ↓
server/stream.py (stream_graph1 或 stream_graph2)
    ↓
graph.stream(inputs, config, stream_mode="updates")
    ↓
各节点按图编排执行:
  Graph1: agent → retrieve ⇄ grade_documents → generate / rewrite
  Graph2: route → retrieve/web_search → grade_documents → generate → grade_hallucinations → grade_answer
    ↓
工具层:
  LLM (deepseek-chat) → 生成/评估
  BGE Embedding → 向量化
  Milvus → 向量检索 (密集+稀疏混合)
  百度 AI 搜索 → 网络搜索兜底
```

---

> **文档生成时间：** 2026-06-26
>
> **项目路径：** `D:\project\pythonProject`
