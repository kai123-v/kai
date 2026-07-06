"""
Markdown 文档解析与切片模块
===========================
负责将 Markdown 文件解析为 LangChain Document 对象，包括：
- 文档加载（使用 Unstructured 库）
- 标题与内容合并（将层级标题附加到内容中）
- 语义切割（对过长内容按语义边界切分）

整个流程:
    Markdown 文件
        │
        ▼ parse_mark_down()
    UnstructuredLoader 加载
    → 按元素分割（标题、段落、列表等）
        │
        ▼ merge_title_content()
    将标题链与对应的内容拼接
    → "一级标题 → 二级标题 → 段落内容"
        │
        ▼ text_chunker()
    SemanticChunker 语义切割
    → 将超过 100 字符的文档按语义边界切分
        │
        ▼
    List[Document] → 写入 Milvus

为什么需要 merge_title_content？
    Markdown 文档中，标题和内容是分开的元素。
    例如:
        # 第一章
        ## 第一节
        这是内容...
    如果不合并，"这是内容..." 这个 Document 的 page_content 就只是 "这是内容..."，
    丢失了它属于哪个章节的信息。
    合并后: "第一章 → 第一节\n这是内容..."
    这样在检索时，即使用户查询的是"第一章"的相关内容，
    这个 Document 也能被匹配到。

为什么需要 text_chunker (语义切割)？
    超长文档（如一个章节几千字）直接作为单个 Document 存入，有多个问题:
    1. Embedding 模型对超长文本的语义编码效果会下降
    2. 检索时返回整个长文档，会撑爆 prompt
    3. 用户问题通常只与文档的某一部分相关
    语义切割能保持语义边界（在自然断点切割），
    相比简单的按字符数切割，质量更高。
"""

from typing import List
import logging

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_unstructured import UnstructuredLoader

from rag.llm_models.embeddings_model import openai_embedding


class MarkdownParser:
    """
    Markdown 文档解析器
    ===================
    将 Markdown 文件解析为结构化的 Document 列表。

    核心组件:
        1. UnstructuredLoader: 文档加载器，将 MD 文件按元素分割
        2. SemanticChunker: 语义切割器，在语义边界处切分长文档
        3. merge_title_content(): 自定义方法，将标题与内容合并
    """

    def __init__(self):
        """
        初始化解析器，配置语义切割器。

        SemanticChunker 参数:
            openai_embedding:
                用于计算文本语义相似度的嵌入模型。
                切割原理: 计算相邻句子的嵌入向量，
                如果相似度低于阈值（意味着语义发生了显著变化），
                就在此处切分。

            breakpoint_threshold_type:
                断点阈值类型，决定如何定义"语义变化显著"。
                可选值:
                    - "percentile": 百分位数（当前使用）
                        取所有相邻句子相似度差异的某个百分位数作为阈值。
                        例如: 如果大部分差异在 0.3-0.5 之间，
                        百分位数阈值会自动适应这个分布。
                        优点: 不需要手动调参数，适应不同文档
                    - "standard_deviation": 标准差
                        以所有相似度差异的均值和标准差为基础。
                    - "interquartile": 四分位差 (IQR)
                        基于相似度差异的四分位数。
                    - "gradient": 梯度
                        基于相似度的变化率（一阶导数）。
        """
        self.text_splitter = SemanticChunker(
            openai_embedding,
            breakpoint_threshold_type="percentile",
        )

    def text_chunker(self, datas: list[Document]) -> List[Document]:
        """
        语义切割方法
        ============
        对长度超过 100 字符的文档进行语义切割。

        参数:
            datas (List[Document]): 待切割的文档列表

        返回:
            List[Document]: 切割后的文档列表（短文档保持原样）

        切割逻辑:
            - 只有 page_content 长度 > 100 字符的文档才会被切割
            - 100 字符以下的短文档（如短段落）保持原样
            - 切割后只保留短块，不保留原始长文档（避免存储重复）
        """
        docs = []  # 存放切割后的文档

        for document in datas: #超过100字的文档进行进行语义切割
            if len(document.page_content) > 100:
                # 条件成立: 内容较长 → 按语义边界切分
                # split_documents() 返回切分后的子文档列表
                # 只保留切分后的短块，不保留原始长文档（避免存储重复）
                docs.extend(self.text_splitter.split_documents([document]))
            else:
                # 短文档不需要切割，直接保留
                docs.append(document)

        return docs

    def markdown_to_documents(self, md_file: str) -> List[Document]:
        """
        Markdown 文件转换为 Document 列表（完整流程）
        ==============================================
        执行完整的三步处理管道: 加载 → 合并 → 切割。

        参数:
            md_file (str):
                Markdown 文件的绝对路径。
                例如: "D:/project/pythonProject/test/test.md"

        返回:
            List[Document]: 处理后的 Document 列表

        处理流程:
            1. parse_mark_down(): 加载并初步分割
            2. merge_title_content(): 合并标题与内容
            3. text_chunker(): 语义切割长文档 → 返回切割后的结果
        """
        # 步骤1: 加载 Markdown 文件
        documents = self.parse_mark_down(md_file)
        logging.info(f"文件解析后的文档数量:{len(documents)}")

        # 步骤2: 合并标题与内容
        merge_documents = self.merge_title_content(documents)
        logging.info(f"合并后的文档数量:{len(merge_documents)}")

        # 步骤3: 语义切割
        chunks_documents = self.text_chunker(merge_documents)
        logging.info(f"切割后的文档数量:{len(chunks_documents)}")

        # 返回切割后的结果（长文档被语义切分，短文档保持原样）
        return chunks_documents

    def parse_mark_down(self, md_file: str) -> List[Document]:
        """
        Markdown 文件加载和元素分割
        ===========================
        使用 Unstructured 库将 MD 文件解析为文档元素。

        参数:
            md_file (str): Markdown 文件路径

        返回:
            List[Document]: 按文档元素分割的 Document 列表

        UnstructuredLoader 参数:
            file_path: 文件路径

            model (str, 默认 'elements'):
                Unstructured 的分割模型。
                可选值:
                    - "elements": 按文档元素分割（标题、段落、列表、代码块等）
                        每个元素成为一个独立的 Document
                        优点: 结构清晰，metadata 丰富
                    - "fast": 快速模式，使用简单的规则分割
                        优点: 速度快，适合大批量处理
                    - "hi_res": 高分辨率模式
                        使用更智能的模型进行分割，效果最好但最慢

            strategy (str, 默认 'fast'):
                分割策略。
                可选值:
                    - "fast": 快速策略（当前使用）
                        使用基本规则快速分割
                        优点: 速度快，资源占用少
                    - "hi_res": 高分辨率策略
                        更准确但更慢
                    - "auto": 自动选择最佳策略
                        根据文档内容自动选择 fast 或 hi_res

        lazy_load() vs load():
            - lazy_load(): 惰性加载，返回生成器
            - load(): 一次性全部加载到内存
            此处使用 lazy_load() 逐个迭代处理，但最终仍收集到列表中，
            因此实际内存占用与 load() 相当。
        """
        loader = UnstructuredLoader(
            file_path=md_file,
            model="elements",  # 按文档元素分割
            strategy="fast",   # 快速模式
        )

        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)

        return docs

    def merge_title_content(self, datas: List[Document]) -> List[Document]:
        """
        将层级标题与对应的内容合并
        ==========================
        Markdown 文档被 Unstructured 分割后，标题和内容是独立的元素。
        此方法将标题链拼接到对应内容中，让每个 Document 自带完整的"章节路径"。

        参数:
            datas (List[Document]): Unstructured 分割后的 Document 列表
                每个 Document 的 metadata 中包含:
                - category: 元素类型 ("Title" 或 "NarrativeText" 等)
                - element_id: 元素唯一标识
                - parent_id: 父元素的 ID（如内容的父元素是标题）

        返回:
            List[Document]: 标题与内容合并后的 Document 列表

        处理逻辑（状态机）:
            遍历所有文档元素，根据 category 分类处理:

            1. category == "NarrativeText" 且 parent_id == "none":
               → 独立内容（没有父标题），直接保留

            2. category == "Title":
               → 这是一个标题元素
               a. 在 metadata 中标记 title（记录标题文本）
               b. 如果它有 parent_id:
                  说明这是子标题（如 ## 的父标题是 #）
                  → 拼接: "父标题内容 → 子标题内容"
               c. 将当前标题记录到 parent_dict（它可能成为后续内容的父级）

            3. category != "Title" 且 parent_id 存在:
               → 这是一个有父标题的内容元素
               a. 拼接: "父标题内容\n自身内容"
               b. 将父标题的 category 改为 "content"（用于后续检索过滤，
                  见 retriever.py 中的 filter={"category": "content"}）
               c. 加入结果列表

            4. 最后: 将 parent_dict 中的标题元素也加入结果
               （因为有些标题可能没有对应的内容，但仍值得保留）

        合并效果示例:
            原始:
                Document1: category=Title, page_content="第一章"
                Document2: category=Title, page_content="第一节", parent_id=id1
                Document3: category=NarrativeText, page_content="这是内容...", parent_id=id2

            合并后:
                Document3: page_content="第一章→第一节\n这是内容..."
                           metadata={"category": "content", "title": "第一章→第一节"}
        """
        merged_datas = []     # 合并后的结果列表
        parent_dict = {}      # 标题映射: {element_id: Document}

        for document in datas:
            metadata = document.metadata

            # 清理不需要的元数据
            if "languages" in metadata:
                metadata.pop("languages")

            # 提取关键元数据
            parent_id = metadata.get("parent_id", None)
            category = metadata.get("category", "none")
            element_id = metadata.get("element_id", "none")

            # ===== 情况1: 独立内容（无父标题）=====
            if category == "NarrativeText" and parent_id == "none":
                # 内容没有关联任何标题 → 直接保留
                merged_datas.append(document)

            # ===== 情况2: 标题元素 =====
            if category == "Title":
                # 在 metadata 中记录标题文本
                document.metadata["title"] = document.page_content

                if parent_id in parent_dict:
                    # 有父标题 → 拼接标题链
                    # "父标题 → 当前标题"
                    document.page_content = (
                        parent_dict[parent_id].page_content
                        + "->"
                        + document.page_content
                    )

                # 将当前标题注册到 parent_dict
                # 后续的内容元素可以通过 parent_id 找到这个标题
                # parent_dict = {
                #     "doc_001": Document(
                #         page_content="这是文档内容...",
                #         metadata={
                #             "category": "other",  # ← 这一层是字典
                #             "source": "file1.pdf",
                #             "author": "张三"
                #         }
                #     )}
                parent_dict[element_id] = document

            # ===== 情况3: 非标题内容，有父标题 =====
            if category != "Title" and parent_id:
                # 拼接: 父标题内容 + 自身内容
                document.page_content = (
                    parent_dict[parent_id].page_content
                    + "\n"
                    + document.page_content
                )

                # 标记父标题的 category 为 "content"
                # 这样在检索时可以通过 filter={"category": "content"} 过滤
                # 不会返回纯标题的 Document（内容为空或太短）。把有内容的二级标题设为content。去掉纯纯标题的
                parent_dict[parent_id].metadata["category"] = "content"

                # 加入结果列表
                merged_datas.append(document)

        # ===== 4. 补充标题元素 =====
        # 将 parent_dict 中所有标题元素也加入结果
        # 有些标题可能有重要信息（如章节标题本身作为查询目标）
        merged_datas.extend(parent_dict.values())

        return merged_datas


# ==================== 测试入口 ====================

if __name__ == "__main__":
    """
    测试 MarkdownParser 的解析效果。
    打印每个 Document 的元数据和内容。
    """
    file_path = "D:/project/pythonProject/test/test.md"
    parser = MarkdownParser()
    docs = parser.markdown_to_documents(file_path)

    for item in docs:
        print(f"元数据:{item.metadata}")
        print(f"标题:{item.metadata.get('title', None)}")
        print(f"doc内容:{item.page_content}\n")
        print("----*" * 10)
