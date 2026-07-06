"""
多进程批量写入 Milvus 模块
===========================
使用生产者-消费者模式，多进程并行处理大量 Markdown 文件的解析和写入。

设计模式: 生产者-消费者 (Producer-Consumer)
    进程1 (生产者 - file_parser_process):
        扫描目录 → 解析 Markdown → 分批放入队列
    进程2 (消费者 - milvus_writer_process):
        从队列读取 → 批量写入 Milvus

为什么需要多进程？
    1. 解析和写入可以并行执行: 一边解析一边写入，不等所有文件解析完
    2. 进程间隔离: 解析失败不会影响写入，反之亦然
    3. 利用多核 CPU: 两个进程可以跑在不同的 CPU 核心上
    4. 内存控制: 队列限制大小，防止解析过快导致内存溢出

为什么用 multiprocessing.Queue 而非 threading.Queue？
    - 多进程 Queue 支持跨进程通信（底层使用管道或共享内存）
    - 多线程 Queue 只能在同一个进程内的线程间通信    所以选多进程。因为是两个进程

终止信号 (Sentinel):
    使用 None 作为特殊的"终止信号"。
    当生产者完成所有文件解析后，向队列放入 None。
    消费者收到 None 后知道没有更多数据了，退出循环。
    这是生产者-消费者模式中常见的"毒丸"模式。
"""

import logging
import multiprocessing
import os.path
from multiprocessing import Queue
from typing import List

from rag.utils.logger import init_logger, get_logger, log
from rag.documents.markdown_parser import MarkdownParser
from rag.documents.milvus_db import MilvusVectorSave


def file_parser_process(dir_path: str, output_queue: Queue, batch_size: int = 20):
    """
    进程1（生产者）: 解析目录下所有 Markdown 文件
    ==============================================
    扫描指定目录，找到所有 .md 文件，解析后分批放入队列。

    参数:
        dir_path (str):
            存放 Markdown 文件的目录路径。
            例如: "D:/project/pythonProject/test"

        output_queue (multiprocessing.Queue):
            进程间通信队列。
            生产者将解析后的 Document 批次放入此队列。
            队列最大容量由调用方设置，防止内存溢出。

        batch_size (int, 默认 20):
            每批包含的 Document 数量。
            当累积的 Document 数量达到此值时，发送一批。
            更大的 batch: 减少队列操作次数，但单批内存占用大
            更小的 batch: 内存更平滑，但队列操作频繁
            当前 20: 在平衡中偏向小批次（防止内存尖峰）

    代码逻辑:
        1. 扫描目录，获取所有 .md 文件的路径
        2. 如果没有 .md 文件 → 发送终止信号并退出
        3. 创建 MarkdownParser 实例
        4. 遍历每个 .md 文件:
           ├── 调用 parser.markdown_to_documents() 解析
           ├── 将解析出的 docs 累积到缓冲区 (doc_batch)
           ├── 缓冲区满 (>= batch_size) → 放入队列 → 清空缓冲区
           └── 解析失败 → 记录错误日志，继续下一个文件
        5. 将缓冲区中剩余的文档放入队列
        6. 发送终止信号 (None)
    """
    logging.info(f"解析进程开始扫描目录：{dir_path}")

    # ===== 1. 扫描目录获取 .md 文件 =====
    md_files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)          # 遍历目录
        if f.endswith(".md")                    # 只取 .md 文件
    ]

    # 如果没有找到任何 .md 文件
    if not md_files:
        log.warning("警告：未找到任何.md文件")
        output_queue.put(None)  # 发送终止信号
        return

    # ===== 2. 创建解析器 =====
    parser = MarkdownParser()
    doc_batch = []  # 文档缓冲区（累积到 batch_size 时发送）

    # ===== 3. 遍历解析每个文件 =====
    for md_path in md_files:
        try:
            # 解析一个 Markdown 文件 → 得到多个 Document
            # 例如一个文件含有 5 个段落，就返回 5 个 Document
            docs = parser.markdown_to_documents(md_path)

            if docs:
                # 将解析出的文档累积到缓冲区
                doc_batch.extend(docs)

            # ===== 4. 缓冲区满 → 发送到队列 =====
            if len(doc_batch) >= batch_size:
                # copy(): 创建副本放入队列
                # 为什么用 copy 而不是直接传引用？  因为传的是指针。而不是对象
                #   因为 clear() 之后，原引用指向的列表会被清空
                #   队列中的数据也会丢失。非
                output_queue.put(doc_batch.copy())
                doc_batch.clear()  # 清空缓冲区，准备下一批

        except Exception as e:
            # 单个文件解析失败 → 记录错误，继续处理下一个文件
            # 为什么继续而不是终止？
            #   一个文件解析失败不应该阻止其他文件的处理
            log.error(f"解析失败 {md_path}: {str(e)}")

    # ===== 5. 处理剩余文档 =====
    # 最后一批可能不满 batch_size，但也要发送
    if doc_batch:
        output_queue.put(doc_batch)

    # ===== 6. 发送终止信号 =====
    # None = "毒丸"，告诉消费者没有更多数据了
    output_queue.put(None)
    log.info(f"解析完成，共处理{len(md_files)}个文件")


def milvus_writer_process(input_queue: Queue):
    """
    进程2（消费者）: 从队列读取文档并写入 Milvus
    ============================================
    从队列中读取 Document 批次，调用 MilvusVectorSave 写入数据库。

    参数:
        input_queue (multiprocessing.Queue):
            进程间通信队列。
            从中读取 Document 批次，收到 None 时退出。

    代码逻辑:
        1. 创建 MilvusVectorSave 实例并建立连接
        2. 循环从队列读取:
           ├── 若为 None → break（终止信号）
           └── 若为 Document 列表 → 写入 Milvus → 累计计数
        3. 打印最终写入总数
    """
    logging.info("milvus写入进程启动")

    # ===== 1. 创建 Milvus 连接 =====
    mv = MilvusVectorSave()
    mv.create_connection()  # 建立 LangChain Milvus 连接

    total_count = 0  # 累计写入计数器

    # ===== 2. 消费循环 =====
    while True:
        # input_queue.get() 是阻塞操作
        # 如果队列为空 → 阻塞等待，直到有数据或收到终止信号
        datas = input_queue.get()

        if datas is None:
            # 条件成立: 收到终止信号 ("毒丸")
            # → 生产者已完成所有工作 → 退出循环
            break

        # 确认是 Document 列表 → 写入 Milvus
        if isinstance(datas, List):
            # add_document 会:
            # 1. 调用 bge_embedding 生成密集向量
            # 2. Milvus 自动触发 BM25 函数生成稀疏向量
            # 3. 写入 text、metadata 等字段
            mv.add_document(datas)

            total_count += len(datas)
            log.info(f"累计已写入:{total_count}个文档")

    log.info(f"写入进程结束,总计写入:{total_count}")


# ==================== 主流程入口 ====================

if __name__ == "__main__":
    """
    批量写入主流程
    ==============
    1. 创建 Milvus 集合
    2. 创建进程间通信队列
    3. 启动两个子进程（生产者 + 消费者）
    4. 等待两个进程完成

    队列大小设置说明:
        maxsize=20 表示队列最多能存放 20 个批次。
        如果生产者速度远快于消费者，队列会填满，
        此时 put() 操作会阻塞，直到消费者取走一个批次。
        这样实现了"背压"机制，防止生产者消耗过多内存。
    """

    # ===== 1. 配置参数 =====
    md_dir = r"D:\project\pythonProject\test"  # MD 文件目录
    queue_maxsize = 20  # 队列最大容量（批次数，不是文档数）

    # ===== 2. 创建 Milvus 集合 =====
    # 注意: create_collection() 需要先于 create_connection() 执行
    # 因为 create_connection() 假设集合已存在
    mv = MilvusVectorSave()
    mv.create_collection()

    # ===== 3. 创建进程间通信队列 =====
    docs_queue = Queue(maxsize=queue_maxsize)

    # ===== 4. 创建子进程 =====
    # 进程1: 生产者 — 解析 MD 文件
    parse_process = multiprocessing.Process(
        target=file_parser_process,
        args=(md_dir, docs_queue),  # 参数传递给 file_parser_process()
    )

    # 进程2: 消费者 — 写入 Milvus
    write_process = multiprocessing.Process(
        target=milvus_writer_process,
        args=(docs_queue,),  # 参数传递给 milvus_writer_process()
    )

    # ===== 5. 启动子进程 =====
    parse_process.start()
    write_process.start()

    # ===== 6. 等待子进程完成 =====
    # join(): 阻塞主进程，直到子进程结束
    parse_process.join()
    write_process.join()

    print("所有进程完成！数据已成功写入 Milvus。")
