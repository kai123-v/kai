"""
百度 AI 搜索工具
================
使用百度千帆 AI 搜索 API 进行联网检索，获取互联网上的最新信息。

为什么需要网络搜索？
    向量知识库的内容是有限的：
    - 只能包含预先导入的文档
    - 无法获取实时信息（新闻、天气、股票等）
    - 知识覆盖范围受限于文档本身
    网络搜索作为"降级方案"，在线索不匹配时提供外部信息源。

@tool 装饰器说明:
    @tool 是 LangChain 的工具注册装饰器。
    它将普通 Python 函数转换为 LangChain Tool 对象：
    - 函数的 docstring 会成为工具的描述（LLM 通过描述了解工具用途）
    - 函数的类型注解会成为工具的输入 schema
    - 函数本身是工具的执行逻辑

API 配置:
    - API Key: 百度千帆平台提供的认证密钥
    - 搜索源: baidu_search_v2（百度搜索 v2 版本）
    - 时效性过滤: month（优先返回最近一个月的内容）
"""

import os
import requests
from langchain_core.tools import tool

# 百度千帆 AI 搜索 API 认证密钥
# 从环境变量 BAIDU_API_KEY 读取
# 获取方式: https://console.bce.baidu.com/ → 千帆大模型平台 → API 密钥管理
API_KEY = os.getenv("BAIDU_API_KEY", "")


@tool
def baidu_search_tool(query: str) -> str:
    """
    使用百度 AI 搜索进行联网搜索，返回搜索结果。
    适用于需要获取最新信息、实时数据或网络资讯的场景。

    参数:
        query (str):
            搜索查询词。通常是将用户问题直接作为查询词。
            LLM/Agent 在调用时自动填充此参数。

    返回:
        str: 百度 AI 搜索 API 的原始响应文本（JSON 格式字符串）。
             包含搜索结果列表，每个结果有标题、摘要、URL 等。

    代码逻辑:
        1. 构造 POST 请求到百度千帆 AI 搜索端点
        2. 设置 Authorization 头（Bearer Token 认证）
        3. 设置搜索参数:
           - messages: 用户查询词
           - search_source: "baidu_search_v2"（使用百度搜索 v2）
           - search_recency_filter: "month"（优先返回最近一个月的内容）
        4. 发起请求
        5. 成功: 返回响应文本
           失败: 返回错误信息

    search_recency_filter 可选值:
        - "day": 优先一天内的内容
        - "week": 优先一周内的内容
        - "month": 优先一月内的内容（当前使用，在时效性和覆盖面上平衡）
        - "year": 优先一年内的内容
        - "none" 或不传: 不进行时效性过滤
    """
    # 百度千帆 AI 搜索的 API 端点
    url = "https://qianfan.baidubce.com/v2/ai_search"

    # 请求头
    headers = {
        # Bearer Token 认证: 将 API Key 放在 Authorization 头中
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # 请求体 - 搜索参数
    data = {
        # messages: 用户查询内容
        # 格式与对话 API 一致: [{"content": "查询内容", "role": "user"}]
        "messages": [{"content": query, "role": "user"}],

        # search_source: 搜索源
        # "baidu_search_v2": 百度搜索 v2 版本，结果质量和时效性更好
        "search_source": "baidu_search_v2",

        # search_recency_filter: 时效性过滤
        # "month": 优先返回最近一个月的内容
        # 可选: "day" | "week" | "month" | "year" | 不传(无过滤)
        "search_recency_filter": "month",
    }

    # 发送 POST 请求
    response = requests.post(url, headers=headers, json=data)

    # 判断请求是否成功
    if response.status_code == 200:
        # HTTP 200: 请求成功 → 返回响应内容
        return response.text
    else:
        # HTTP 非 200: 请求失败 → 返回错误信息
        # 注意: 返回的是字符串而非抛出异常，因为这是工具函数，
        # 返回错误字符串可以让 LLM 知道搜索失败并尝试其他方式
        return f"搜索失败: {response.status_code}"
