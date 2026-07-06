"""
环境变量 / 全局配置
===================
集中管理项目中使用的常量配置。

配置项说明:

    MILVUS_URI (str):
        Milvus 向量数据库的连接地址。
        格式: http://{host}:{port}
        当前: localhost:19530
        - 19530 是 Milvus 的默认端口
        - 如果 Milvus 部署在其他机器上，改为对应 IP 地址
        - 如果使用 Milvus Cloud 或 Zilliz Cloud，改为云服务地址

    COLLECTION_NAME (str):
        Milvus 中的集合（表）名称。
        - 相当于关系数据库中的"表"
        - 所有文档都存储在同一个 collection 中
        - 不同的 collection 可以用于不同的知识库（如 technical_docs, faq 等）
        - 当前: "rag_table"
"""

# Milvus 服务连接地址
# 可选值示例:
#   - "http://localhost:19530" — 本地开发环境
#   - "http://192.168.1.100:19530" — 局域网内的 Milvus 服务器
#   - "https://your-instance.zillizcloud.com:19530" — 云服务
MILVUS_URI = "http://localhost:19530"

# Milvus 集合名称
# 不同项目/知识库应使用不同的 collection 名，避免数据混乱
COLLECTION_NAME = "rag_table"
