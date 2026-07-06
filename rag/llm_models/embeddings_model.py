"""
LLM 和 Embedding 模型配置模块
=============================
集中管理所有 AI 模型的配置，包括：
- Embedding 模型：用于将文本转换为向量
- LLM（大语言模型）：用于生成、评估、路由等所有文本处理任务

为什么需要两个 Embedding 模型？
    1. openai_embedding: 通过 OpenAI 兼容 API 调用（GemAI 代理）
       用于 MarkdownParser 中的 SemanticChunker（语义切割器）
       需要外部 API 调用，速度较慢但语义理解能力更强
    2. bge_embedding: 本地运行的开源中文 Embedding 模型
       用于 Milvus 向量库的文档向量化和查询向量化
       本地运行，速度快，中文语义效果好，不消耗 API 额度

当前使用的 LLM:
    - 模型: deepseek-chat (通过 GemAI API 代理)
    - 特点: 性价比高，中文能力强，支持 function calling
    - 温度: 0.5 → 在创造性和确定性之间取平衡
        温度范围 0.0 ~ 2.0：
        - 0.0: 完全确定性，每次输出相同（适合评分、评估等需要一致性的任务）
        - 0.5: 轻度创造性（当前使用，适合一般对话）
        - 1.0+: 高创造性，输出变化大（适合创意写作）
        - 2.0: 最大创造性（输出可能不稳定）
"""

import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ==================== API 配置 ====================
# 从环境变量读取，避免密钥泄露
# 本地开发时在项目根目录创建 .env 文件，或在系统环境变量中设置:
#   OPENAI_API_KEY=sk-xxx
#   OPENAI_API_BASE=https://api.gemai.cc/v1
#   OPENAI_EMBEDDING_MODEL=[官逆]gpt-4o-mini
#   LLM_MODEL=deepseek-chat

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.gemai.cc/v1")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "[官逆]gpt-4o-mini")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# ==================== OpenAI 兼容 Embedding ====================
# 通过 GemAI API 代理调用 OpenAI 兼容的 Embedding 接口
# 主要用于 MarkdownParser 的 SemanticChunker（语义切割器）
openai_embedding = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE,
    model=OPENAI_EMBEDDING_MODEL,
)

# ==================== 本地 Embedding (BGE) ====================
# 使用 HuggingFace 本地加载 BAAI/bge-small-zh-v1.5
# 这是中文领域最流行的开源 Embedding 模型之一
# 特点: 512 维向量，轻量快速，中文语义效果好

# model_name: 模型标识符
# "BAAI/bge-small-zh-v1.5":
#   - BAAI: 北京智源人工智能研究院
#   - bge: BAAI General Embedding（通用嵌入模型系列）
#   - small: 小型版本（还有 base、large 等）
#   - zh: 针对中文优化
#   - v1.5: 版本号
# 首次加载时会自动从 HuggingFace 下载模型（约 100MB）
model_name = "BAAI/bge-small-zh-v1.5"

# model_kwargs: 模型运行参数
# device: 指定模型运行在哪个设备上
#   可选值:
#     - "cpu": 在 CPU 上运行（当前使用）
#         优点: 兼容性最好，无需 GPU
#         缺点: 速度较慢
#     - "cuda": 在 NVIDIA GPU 上运行
#         优点: 速度快（5-10x CPU）
#         缺点: 需要 CUDA 环境和 GPU 显存
#     - "cuda:0" / "cuda:1": 指定特定 GPU
#     - "mps": 在 Apple Silicon GPU 上运行（M1/M2/M3 Mac）
model_kwargs = {"device": "cpu"}

# encode_kwargs: 向量编码参数
# normalize_embeddings: 是否对向量进行 L2 归一化
#   True:
#     - 将向量长度归一化为 1
#     - 归一化后，向量内积 (Inner Product) = 余弦相似度 (Cosine Similarity)
#     - 推荐开启，因为：
#       1. Milvus 使用 IP (Inner Product) 作为相似度度量
#       2. 归一化后 IP 等价于余弦相似度
#       3. 余弦相似度比原始 IP 更合理（不受向量长度影响）
#   False:
#     - 保留原始向量长度
#     - 适用于某些不需要归一化的场景
encode_kwargs = {"normalize_embeddings": True}

# 创建 BGE Embedding 实例
bge_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)


# ==================== LLM 配置（大语言模型） ====================
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE,
    model=LLM_MODEL,

    # temperature: 生成温度（控制输出的随机性/创造性）
    # 取值范围: 0.0 ~ 2.0
    #   0.0: 完全确定性输出，每次相同输入得到相同结果
    #        - 适合: 评分、评估、路由、分类等任务
    #        - 缺点: 回答可能显得机械化
    #   0.5: 轻度创造性（当前使用）
    #        - 适合: 一般对话问答、内容生成
    #        - 平衡了准确性和自然性
    #   1.0: 较高创造性
    #        - 适合: 创意写作、头脑风暴
    #   2.0: 最大随机性
    #        - 输出可能不稳定，通常不推荐
    # 为什么选 0.5？
    #   这个项目既需要准确性（评估、路由）又需要自然（生成回答），
    #   0.5 在两个目标之间取得了平衡。
    temperature=0.5,
)
