"""
Milvus 向量数据库连接与管理模块
===============================
负责 Milvus 集合(Collection)的创建、连接和文档的增删改查。

什么是 Milvus？
    Milvus 是一个开源的向量数据库，专门用于存储和检索高维向量数据。
    在 RAG 系统中，它的作用类似"语义搜索引擎"：
    - 存储: 将文档转换为向量后存入
    - 检索: 将问题转换为向量后搜索最相似的文档

混合检索 (Hybrid Search):
    本项目使用了两种向量表示：
    1. 密集向量 (Dense Vector):
       - 由 BGE Embedding 模型生成（512 维浮点数向量）
       - 捕获语义含义（"AI" 和 "人工智能" 在语义上接近）
       - 使用 HNSW 索引，IP (Inner Product) 相似度
    2. 稀疏向量 (Sparse Vector):
       - 由 BM25 算法生成（维度 = 词汇表大小）
       - 捕获关键词匹配（精确匹配 "神经网络" 这个词）
       - 使用 SPARSE_INVERTED_INDEX

    两者通过 RRF (Reciprocal Rank Fusion) 算法融合排序，
    兼顾了语义理解和关键词匹配。

Milvus 与 LangChain 的关系:
    - PyMilvus (pymilvus): Milvus 官方 Python SDK，底层操作
    - LangChain-Milvus (langchain_milvus): LangChain 的 Milvus 集成，
      封装了 PyMilvus，提供与 LangChain Document/Retriever 的无缝连接

同一份文档的内容做成两个嵌入模型向量字段存入数据库。他还有其他字段。比如id或者创建时间。比如一条数据信息有id 稀疏向量字段 密集向量字段
"""

from typing import List
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import MilvusClient, connections, Function
from pymilvus.client.types import MetricType, IndexType, DataType, FunctionType

from rag.documents.markdown_parser import MarkdownParser
from rag.llm_models.embeddings_model import bge_embedding
from rag.utils.env_utils import MILVUS_URI, COLLECTION_NAME


class MilvusVectorSave:
    """
    Milvus 向量数据库管理器
    =======================
    负责集合的创建、连接和文档的插入。

    两个核心对象:
        1. self.client (pymilvus.MilvusClient):
           底层 PyMilvus 客户端，用于管理操作（创建集合、建索引等）
        2. self.vector_store_saved (langchain_milvus.Milvus):
           LangChain 的 Milvus 包装，用于文档操作（增删查）
           内部也包含一个 client，两者连接到同一个 Milvus 服务

    为什么需要两个客户端？
        - PyMilvus 客户端: 自定义 Schema 和索引，灵活性高
        - LangChain Milvus: 提供与 LangChain 生态的无缝集成
        （retriever、Document、embedding_function 等）
    """

    def __init__(self):
        """
        初始化 MilvusVectorSave 实例。
        vector_store_saved 初始为 None，由 create_connection() 设置。
        """
        self.vector_store_saved = None

    def create_collection(self):
        """
        创建 Milvus 集合（自定义 Schema）
        ================================
        使用 PyMilvus 底层 API 创建集合，自定义字段和索引。

        为什么不直接用 LangChain 的 Milvus 来创建集合？
            LangChain Milvus 创建的集合字段是固定的（id + vector + text）。
            但本项目需要：
            - 双向量字段（dense + sparse）→ 混合检索
            - 自定义元数据字段（category, source, filename 等）
            - BM25 自动编码函数
            - 中文分词器 (jieba)
            这些功能 LangChain 的默认集合创建不提供。

        如果集合已存在怎么办？
            先释放 → 删除索引 → 删除集合 → 重新创建。
            这是为了确保 Schema 与代码一致（适合开发阶段）。
            生产环境可能需要保留数据，应该用"如果已存在则跳过"的逻辑。

        Milvus Schema 字段说明:
            id (INT64): 主键，自增
                - 唯一标识每个文档
                - auto_id=True: 由 Milvus 自动分配，不需要手动设置

            text (VARCHAR 6000): 文档文本内容
                - max_length=6000: 最多 6000 个字符
                - enable_analyzer=True: 启用分词器
                - tokenizer="jieba": 使用结巴分词器（中文分词）
                - filter=["cnalphanumonly"]: 过滤掉标点符号，只保留中文和字母数字

            category (VARCHAR): 文档类别
                - 用于过滤检索结果，如 filter={"category": "content"}
                - 可选值: "content"（内容文档）, "Title"（标题文档）

            source, filename, filetype, title (VARCHAR): 文档元数据
                - 记录文档的来源信息，方便追溯

            category_depth (INT64): 类别深度
                - 标题的层级深度（1=一级标题，2=二级标题...）

            sparse (SPARSE_FLOAT_VECTOR): 稀疏向量
                - 由 BM25 函数从 text 字段自动生成
                - 用于关键词匹配

            dense (FLOAT_VECTOR, dim=512): 密集向量
                - 由 BGE Embedding 模型从 text 字段生成
                - 维度 512（与 bge-small-zh-v1.5 的输出匹配）
                - 用于语义匹配

        索引配置:
            稀疏向量索引 (SPARSE_INVERTED_INDEX):
                - 用于 BM25 稀疏向量的快速检索
                - metric_type="BM25": 使用 BM25 相似度度量(其实是关键词匹配)  评判搜出来的向量分数标准      计算向量与向量计算的方式。问题向量与答案向量
                - inverted_index_algo="DAAT_MAXSCORE": 倒排索引算法                 怎么搜索更快
                  DAAT = Document-At-A-Time (按文档处理)
                  MAXSCORE = 最大分数剪枝（加速查询）
                - bm25_k1=1.2: BM25 词频饱和参数（1.2~2.0 之间）
                  控制词频对相关性的影响程度            重复多少遍算够
                - bm25_b=0.75: BM25 文档长度归一化参数（0~1 之间）
                  0=完全按长度归一化，1=完全不归一化     长度惩罚。是否太长会高分

            密集向量索引 (HNSW):
                - HNSW = Hierarchical Navigable Small World
                  一种基于图的近似最近邻搜索算法
                - 特点: 查询速度快，精度高，但内存占用较高
                - 替代: IVF_FLAT（倒排+聚类，内存低但精度略低）
                       IVF_SQ8（量化版 IVF，内存极低但有精度损失）
                - M=16: 每个节点的最大连接数
                  越大: 精度越高，但索引越大
                - efConstruction=64: 构建时的搜索范围
                  越大: 索引质量越好，但构建越慢
                - metric_type=IP: Inner Product (内积)
                  因为 BGE embedding 启用了归一化，所以 IP = 余弦相似度    注意：bge_embedding 在 embeddings_model.py 中已设置 normalize_embeddings=True
        """
        # 连接 Milvus
        client = MilvusClient(uri=MILVUS_URI)

        # 创建 Schema
        # enable_dynamic_field=True: 允许动态添加字段（如自动存储 metadata 中的字段）
        # 这样 LangChain 传入的 metadata 中的字段（如 parent_id）会被自动存储
        schema = client.create_schema(enable_dynamic_field=True)

        # ===== 添加字段 =====

        # 主键: 唯一标识
        schema.add_field(
            field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True
        )

        # 文本内容: 核心字段，启用中文分词
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=6000,
            enable_analyzer=True,
            # analyzer_params: 分词器设置
            # tokenizer="jieba": 使用结巴分词器
            # filter=["cnalphanumonly"]: 只保留中文和字母数字，去掉标点
            #   等效于 analyzer_params={"type": "chinese"} 的简写
            analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]},
        )

        # ===== 元数据字段 =====
        schema.add_field(
            field_name="category", datatype=DataType.VARCHAR, max_length=1000,
            nullable=True
        )
        schema.add_field(
            field_name="source", datatype=DataType.VARCHAR, max_length=1000,
            nullable=True
        )
        schema.add_field(
            field_name="filename", datatype=DataType.VARCHAR, max_length=1000,
            nullable=True
        )
        schema.add_field(
            field_name="filetype", datatype=DataType.VARCHAR, max_length=1000,
            nullable=True
        )
        schema.add_field(
            field_name="title", datatype=DataType.VARCHAR, max_length=1000,
            nullable=True
        )
        schema.add_field(
            field_name="category_depth", datatype=DataType.INT64, nullable=True
        )

        # ===== 向量字段 =====
        # 稀疏向量: BM25 自动生成
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 密集向量: BGE Embedding 生成
        # dim=512 必须与 bge-small-zh-v1.5 的输出维度一致
        schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=512)

        # ===== BM25 函数 =====
        # Function: Milvus 内置函数，自动将 text 字段转换为 sparse 向量
        # 当文档插入时，Milvus 自动执行此函数
        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],  # 输入: text 字段
            output_field_names=["sparse"],  # 输出: sparse 字段
            function_type=FunctionType.BM25,  # BM25 编码
        )
        schema.add_function(bm25_function)

        # ===== 创建索引 =====
        index_params = client.prepare_index_params()

        # 稀疏向量索引 (BM25)
        index_params.add_index(
            field_name="sparse",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",  # 倒排索引
            metric_type="BM25",                   # BM25 相似度  本质是关键词匹配
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",  # 倒排索引算法
                "bm25_k1": 1.2,  # 词频饱和参数（控制词频的贡献度）
                "bm25_b": 0.75,  # 文档长度归一化参数
            },
        )

        # 密集向量索引 (HNSW)
        index_params.add_index(
            field_name="dense",
            index_name="dense_inverted_index",
            index_type=IndexType.HNSW,  # HNSW 图算法
            metric_type=MetricType.IP,  # Inner Product (内积)
            params={
                "M": 16,              # 每个节点的最大连接数
                "efConstruction": 64,  # 构建时的搜索范围
            },
        )

        # ===== 删除旧集合（如果存在）=====
        # 这是为了确保 Schema 与代码一致
        # 生产环境建议改为"如果已存在则跳过"
        if COLLECTION_NAME in client.list_collections():
            client.release_collection(collection_name=COLLECTION_NAME)
            client.drop_index(
                collection_name=COLLECTION_NAME, index_name="dense_inverted_index"
            )
            client.drop_index(
                collection_name=COLLECTION_NAME, index_name="sparse_inverted_index"
            )
            client.drop_collection(collection_name=COLLECTION_NAME)

        # 创建新集合
        client.create_collection(
            collection_name=COLLECTION_NAME, schema=schema, index_params=index_params
        )

        # 保存 client 引用
        self.client = client

    def create_connection(self):
        """
        创建 LangChain Milvus 连接
        ==========================
        使用 LangChain 的 Milvus 封装创建连接，用于：
        - 将 LangChain Document 存入 Milvus
        - 创建 Retriever（检索器）
        - 与 LangGraph/LangChain 生态无缝集成

        关键参数说明:

            embedding_function (bge_embedding):
                用于生成密集向量的嵌入函数。
                当调用 add_documents() 时，LangChain 自动调用此函数生成向量。

            collection_name (COLLECTION_NAME):
                要连接的 Milvus 集合名称。
                必须与 create_collection() 中创建的集合名称一致。

            builtin_function (BM25BuiltInFunction()):
                LangChain 的 BM25 内置函数封装。
                告诉 LangChain: 除了密集向量，还要用 BM25 生成稀疏向量。

            vector_field (["dense", "sparse"]):
                指定向量存储的字段名。
                两个字段: dense (密集向量) + sparse (稀疏向量) = 混合检索

            consistency_level ("Strong"):
                数据一致性级别。
                可选值:
                    - "Strong": 强一致性
                        写入后立即对所有查询可见
                        优点: 数据绝对一致，不会读到脏数据
                        缺点: 写入性能最差
                        适合: 需要数据绝对准确的场景（如测试环境）
                    - "Session": 会话一致性
                        同一连接内保证一致性
                        适合: 用户会话场景（用户写入后立即查询）
                    - "Bounded": 有限滞后一致性（默认）
                        在指定时间窗口内保证一致性
                        适合: 大多数生产场景，性能与一致性平衡
                    - "Eventually": 最终一致性
                        最终会一致，但没有时间保证
                        优点: 写入最快
                        缺点: 可能读到旧数据
                        适合: 日志、评论等对一致性不敏感的场景

            auto_id (True):
                是否自动生成主键 ID。
                True: Milvus 自动分配 ID（不重复）
                False: 需要手动指定 ID

            enable_dynamic_field (True):
                是否支持动态字段。
                True: metadata 中的字段（如 parent_id）会自动加入 schema，
                      不需要预先定义
                False: 只接受 schema 中明确定义的字段

            connection_args ({"uri": MILVUS_URI}):
                Milvus 服务连接参数。
        """
        # 创建底层 PyMilvus 客户端
        client = MilvusClient(uri=MILVUS_URI)

        # 解析 URI 获取主机名和端口
        parsed = urlparse(MILVUS_URI)

        # connections.connect(): PyMilvus 的连接管理
        # alias: 连接的别名（用于多连接场景）
        # host: Milvus 服务主机
        # port: Milvus 服务端口
        connections.connect(
            alias=client._using, host=parsed.hostname, port=str(parsed.port)
        )

        # 创建 LangChain Milvus 实例
        self.vector_store_saved = Milvus(
            embedding_function=bge_embedding,       # 密集向量生成器  密集向量要调用外部嵌入模型计算
            collection_name=COLLECTION_NAME,        # 集合名称
            builtin_function=BM25BuiltInFunction(), # BM25 稀疏向量函数    milvus内置bm25算法。不需要调用
            vector_field=["dense", "sparse"],       # 双向量字段
            consistency_level="Strong",             # 强一致性
            auto_id=True,                           # 自动生成 ID
            enable_dynamic_field=True,              # 支持动态字段
            connection_args={"uri": MILVUS_URI},    # 连接地址
        )

    def add_document(self, datas: List[Document]):
        """
        将文档批量写入 Milvus
        =====================
        接收 LangChain Document 列表，调用 vector_store_saved.add_documents() 写入。

        参数:
            datas (List[Document]):
                要写入的文档列表。每个 Document 包含:
                - page_content (str): 文档文本内容 → 存入 text 字段
                - metadata (dict): 文档元数据 → 存入对应字段

        特殊处理:
            metadata 中的 "languages" 字段如果是列表，转换为逗号分隔的字符串。
            原因: Milvus 的 VARCHAR 字段不支持列表类型，
                  需要序列化为字符串才能存储。

        代码逻辑:
            1. 调用 vector_store_saved.add_documents() 批量写入
            2. LangChain 会自动调用 embedding_function 生成密集向量
            3. Milvus 的内置 BM25 函数自动生成稀疏向量
        """
        # 批量写入
        # LangChain 会自动:
        # 1. 调用 bge_embedding 生成 512 维密集向量 → 写入 dense 字段
        # 2. Milvus 自动触发 BM25 函数 → 写入 sparse 字段
        # 3. page_content → 写入 text 字段
        # 4. metadata 各字段 → 写入对应字段
        self.vector_store_saved.add_documents(datas)


# ==================== 测试入口 ====================

if __name__ == "__main__":
    """
    测试流程:
    1. 解析 Markdown 文件 → Document 列表
    2. 创建 Milvus 集合
    3. 建立连接
    4. 写入文档
    5. 验证: 查看集合信息和索引信息
    """
    file_path = "D:/project/pythonProject/test/test.md"

    # 步骤1: 解析文档
    parser = MarkdownParser()
    docs = parser.markdown_to_documents(file_path)

    # 步骤2-4: 写入 Milvus
    milvus = MilvusVectorSave()
    milvus.create_collection()   # 创建集合 + 索引
    milvus.create_connection()   # 建立 LangChain 连接
    milvus.add_document(docs)    # 写入文档

    # 步骤5: 验证
    client = milvus.vector_store_saved.client

    # 查看集合描述
    desc_collection = client.describe_collection(collection_name=COLLECTION_NAME)

    # 查看所有索引
    res = client.list_indexes(collection_name=COLLECTION_NAME)
    print("表中所有索引：", res)

    # 查看每个索引的详细描述
    if res:
        for i in res:
            desc_index = client.describe_index(
                collection_name=COLLECTION_NAME, index_name=i
            )
            print(desc_index)
