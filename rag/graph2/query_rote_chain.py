"""
Graph2 — 问题路由链
===================
定义问题路由的 LLM 链，用于判断用户问题应该：
- 走向量知识库检索（vectorstore）
- 还是走网络搜索（web_search）

这是 Graph2 的"入口守卫"，在流程开始时就决定检索策略。

为什么需要路由？
    不是所有问题都适合用向量库检索：
    - "什么是反向传播？" → 向量库中有 AI 知识 → vectorstore
    - "今天天气怎么样？" → 向量库中没有天气数据 → web_search
    路由可以避免浪费向量检索的资源，直接把不合适的问题导向网络搜索。
"""


from langchain_core.prompts import ChatPromptTemplate
from typing import Literal
from pydantic import BaseModel
from pydantic import Field

from rag.llm_models.embeddings_model import llm


class RouteQuery(BaseModel):
    """
    问题路由决策的结构化输出模型
    ===========================
    要求 LLM 必须按此格式返回路由决策。

    datasource (Literal["vectorstore", "web_search"]):
        数据源选择。
        为什么用 Literal 类型？
            Literal 限制了 LLM 只能输出这两个值，不能输出其他内容。
            Pydantic 的 Field(description=...) 会作为 LLM 的提示，
            告诉 LLM 这个字段的含义和可选值。

        "vectorstore": 走 Milvus 向量知识库检索
            适用场景: AI、机器学习、深度学习、数学基础等相关主题
            特点: 检索速度快，内容可控，但知识有限

        "web_search": 走百度 AI 搜索联网检索
            适用场景: 向量库不包含的领域知识、实时信息、新闻等
            特点: 信息最新最全，但速度较慢，内容不可控
    """
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,  # ... 表示必填，没有默认值
        description="根据用户问题选择将其路由到向量知识库或者网络搜索"
    )


# ===== 创建带结构化输出的 LLM =====
# with_structured_output() 将 Pydantic 模型注入 LLM 的 function calling
# method="function_calling":
#   使用 OpenAI 兼容的 function calling 机制
#   告诉 LLM: "你只能调用一个函数来返回结果，该函数需要 datasource 参数"
#   可选值: "function_calling" | "json_mode" | "json_schema"
#   - "function_calling": 通过 tool/function calling 返回结构化数据（最常用）
#   - "json_mode": 强制 LLM 输出纯 JSON（简单场景）
#   - "json_schema": 使用结构化 JSON schema（更严格的约束）
structured_llm_router = llm.with_structured_output(
    RouteQuery, method="function_calling"
)

# ===== 路由提示词 =====
# ChatPromptTemplate.from_messages 构建多轮对话格式的提示词
# ("system", text): 系统角色消息，设定 LLM 的行为准则
# ("human", "{question}"): 用户角色消息，{question} 是占位符
system = """你是一个擅长将用户问题路由到向量知识库或网格搜索的专家。
向量知识库包含关于人工智能、机器学习、深度学习、数学基础等相关知识
对于这些主题的问题请使用向量知识库，其他情况使用网格搜索
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# ===== 构建问题路由链 =====
# | 管道运算符:
#   route_prompt: 将 {"question": "..."} 格式化为提示词消息
#   structured_llm_router: LLM 处理并返回 RouteQuery 对象
#
# 调用示例:
#   result = question_route_chain.invoke({"question": "什么是机器学习？"})
#   print(result.datasource)  # → "vectorstore"
#   result = question_route_chain.invoke({"question": "今天天气怎么样？"})
#   print(result.datasource)  # → "web_search"
question_route_chain = route_prompt | structured_llm_router
