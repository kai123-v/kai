"""
日志工具模块
============
提供统一的日志管理，支持：
- 控制台 + 文件双输出
- 多进程环境（每个进程独立的日志记录）
- 按日期自动轮转（RotatingFileHandler）
- 单例模式（同一 name 只创建一个实例）

日志级别说明（从低到高）:
    DEBUG (10):   详细调试信息，开发时使用
    INFO (20):    正常流程信息（节点执行、数据统计等）— 默认级别
    WARNING (30): 警告信息（异常但可恢复，如 LLM 调用失败后降级）
    ERROR (40):   错误信息（需要关注的问题）
    CRITICAL (50):严重错误（系统级故障）

日志格式说明:
    '%(asctime)s - %(processName)s - %(process)d - %(name)s - %(levelname)s - %(message)s'
    示例:
    2026-06-26 21:30:15 - MainProcess - 12345 - default - INFO - 调用智能体

单例模式说明:
    单例模式确保同一个 name 只创建一个 Logger 实例。
    例如:
        log1 = Logger(name='app')  # 创建实例
        log2 = Logger(name='app')  # 返回同一个实例（不创建新的）
    好处:
        - 避免重复创建 handler（控制台和文件处理器）
        - 确保同一名称的日志设置一致
        - 节省资源
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler


class Logger:
    """
    日志工具类
    ==========
    封装 Python 标准 logging 库，提供便捷的日志输出功能。

    核心特性:
        1. 单例模式: 同一 name 共享实例（_instances 字典管理）
        2. 双输出: 同时输出到控制台和文件
        3. 按日轮转: 日志文件按天命名，自动归档旧日志
        4. 大小轮转: 单文件超过 max_bytes 时自动切割
        5. 多进程兼容: 每个进程独立 Logger 实例
    """

    # 类变量 — 所有实例共享
    _instances = {}           # 存储所有已创建的 Logger 实例（按 name 索引）
    _default_level = logging.INFO  # 默认日志级别

    def __new__(cls, name='app', log_dir='logs', level=logging.INFO,
                console=True, file=True, max_bytes=10 * 1024 * 1024, backup_count=5):
        """
        单例模式 — 控制实例创建
        ========================
        如果 name 已存在实例，返回已有实例（不重新创建）。
        如果 name 不存在，创建新实例并注册到 _instances。

        参数:
            name (str, 默认 'app'):
                日志器名称，用于区分不同的日志器。
                同一 name 共享实例，不同 name 创建独立实例。
                示例: 'default'、'worker_0'、'custom'

        Returns:
            Logger 实例（新创建的或已有的）
        """
        if name not in cls._instances:
            # 该 name 的实例还不存在 → 创建新的
            instance = super().__new__(cls)
            instance._initialized = False  # 标记为未初始化
            cls._instances[name] = instance
        # 返回已有（或刚创建）的实例
        return cls._instances[name]

    def __init__(self, name='app', log_dir='logs', level=logging.INFO,
                 console=True, file=True, max_bytes=10 * 1024 * 1024, backup_count=5):
        """
        初始化日志器
        ============
        如果已经初始化过（_initialized == True），跳过不重复初始化。
        这保证了单例模式下，多次调用 __init__ 不会重置配置。

        参数:
            name (str, 默认 'app'):
                日志器名称。
                会显示在日志格式中的 %(name)s 位置。
                多进程场景中，每个进程用不同的 name。

            log_dir (str, 默认 'logs'):
                日志文件存放目录。
                - 默认在项目根目录的 logs/ 下
                - 如果目录不存在，自动创建

            level (int, 默认 logging.INFO):
                日志级别。
                可选值及其含义:
                    - logging.DEBUG (10):   输出所有级别日志（最详细）
                    - logging.INFO (20):    输出 INFO 及以上级别
                    - logging.WARNING (30): 输出 WARNING 及以上级别
                    - logging.ERROR (40):   输出 ERROR 及以上级别
                    - logging.CRITICAL (50): 仅输出严重错误

            console (bool, 默认 True):
                是否输出到控制台（终端）。
                - True: 同时输出到控制台和文件
                - False: 仅输出到文件（适合后台运行的生产环境）

            file (bool, 默认 True):
                是否输出到文件。
                - True: 写入日志文件
                - False: 不写文件（仅控制台输出，适合临时调试）

            max_bytes (int, 默认 10*1024*1024 = 10MB):
                单个日志文件的最大大小（字节）。
                超过此大小时，自动创建新文件，旧文件被重命名（轮转）。
                示例值:
                    - 1048576 (1MB): 适合高频日志场景
                    - 10485760 (10MB): 当前使用，适合一般场景
                    - 104857600 (100MB): 适合低频日志场景

            backup_count (int, 默认 5):
                保留的备份文件数量。
                旧的备份文件命名: name_20260626.log.1, name_20260626.log.2, ...
                超过此数量的最旧文件会被删除。
                示例: 5 表示保留最近 5 个备份文件。
        """
        # 如果已初始化 → 跳过（单例模式保护）
        if self._initialized:
            return

        # 保存配置
        self.name = name
        self.log_dir = log_dir
        self.level = level
        self.console = console
        self.file = file
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        # ===== 创建底层 logging.Logger =====
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # 清除已有 handler（避免重复添加，如单例模式下多次初始化）
        if self.logger.handlers:
            self.logger.handlers.clear()

        # ===== 设置日志格式 =====
        self.formatter = logging.Formatter(
            # 格式说明:
            #   %(asctime)s: 时间戳（如 2026-06-26 21:30:15）
            #   %(processName)s: 进程名称（如 MainProcess, Worker-0）
            #   %(process)d: 进程 ID（PID）
            #   %(name)s: 日志器名称
            #   %(levelname)s: 日志级别（INFO, WARNING 等）
            #   %(message)s: 日志消息正文
            "%(asctime)s - %(processName)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 按需添加 handler
        if console:
            self._add_console_handler()  # 添加控制台输出
        if file:
            self._add_file_handler()     # 添加文件输出

        self._initialized = True  # 标记已初始化

    def _add_console_handler(self):
        """
        添加控制台处理器
        ================
        将日志输出到 sys.stdout（终端）。
        控制台 handler 使用与文件相同的格式和级别。
        """
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

    def _add_file_handler(self):
        """
        添加文件处理器
        ==============
        将日志输出到文件，支持按日期和按大小轮转。

        文件命名规则:
            {name}_{YYYYMMDD}.log
            例如: default_20260626.log

        轮转机制:
            RotatingFileHandler 在文件达到 max_bytes 时自动轮转：
            1. 关闭当前文件
            2. 将当前文件重命名为 .log.1
            3. 将已有的 .log.1 重命名为 .log.2
            4. 依次类推（最多保留 backup_count 个备份）
            5. 创建新的空日志文件
        """
        # 确保日志目录存在
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 构建日志文件路径
        # 文件名包含日期: default_20260626.log
        # 不同日期的日志自动分离
        log_file = os.path.join(
            self.log_dir,
            f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log",
        )

        # 创建轮转文件 handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.max_bytes,       # 单文件最大字节数
            backupCount=self.backup_count,  # 保留的备份数量
            encoding="utf-8",               # UTF-8 编码支持中文
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

    # ===== 日志输出方法 =====
    # 每个方法对应一个日志级别

    def get_logger(self):
        """
        获取底层 logging.Logger 实例。
        用于需要直接使用标准 logging API 的场景。
        """
        return self.logger

    def debug(self, message):
        """输出 DEBUG 级别日志 — 详细调试信息"""
        self.logger.debug(message)

    def info(self, message):
        """输出 INFO 级别日志 — 正常流程信息"""
        self.logger.info(message)

    def warning(self, message):
        """输出 WARNING 级别日志 — 可恢复的异常"""
        self.logger.warning(message)

    def error(self, message):
        """输出 ERROR 级别日志 — 需要关注的错误"""
        self.logger.error(message)

    def critical(self, message):
        """输出 CRITICAL 级别日志 — 严重系统故障"""
        self.logger.critical(message)

    def exception(self, message):
        """
        输出异常日志。
        与 error() 的区别: 自动附加当前异常的堆栈跟踪信息。
        通常用在 except 块中。
        """
        self.logger.exception(message)


# ==================== 便捷函数 ====================

def get_logger(name='app', log_dir='logs', level=logging.INFO):
    """
    获取 Logger 实例（便捷函数）
    ============================
    简化 Logger 的创建，使用默认参数即可快速获取日志器。

    参数:
        name: 日志器名称
        log_dir: 日志目录
        level: 日志级别

    Returns:
        Logger 实例

    示例:
        log = get_logger('my_module', 'logs', logging.DEBUG)
        log.info("这条日志会显示")
    """
    return Logger(name=name, log_dir=log_dir, level=level)


def init_logger(name='app', log_dir='logs', level=logging.INFO, console=True, file=True):
    """
    初始化日志器并返回底层 logging.Logger（用于多进程等场景）
    ==========================================================
    与 get_logger 类似，但返回的是标准 logging.Logger 对象，
    可以在多进程环境中使用。

    示例（多进程）:
        def worker(worker_id):
            worker_log = init_logger(f'worker_{worker_id}', 'logs')
            worker_log.info(f"工作进程 {worker_id} 启动")
    """
    logger = Logger(name=name, log_dir=log_dir, level=level, console=console, file=file)
    return logger.get_logger()


# ==================== 默认日志实例 ====================

# 创建一个全局默认日志实例，供其他模块直接导入使用
# 用法:
#   from rag.utils.logger import log
#   log.info("这是一条日志")
#   log.warning("这是一条警告")
log = get_logger('default', 'logs', logging.INFO)


# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 示例1: 直接使用全局 log 对象
    log.info("这是默认日志")
    log.warning("这是警告日志")

    # 示例2: 创建自定义日志器（不同级别）
    custom_log = init_logger('custom', 'logs', logging.DEBUG)
    custom_log.debug("这是 DEBUG 日志，默认级别的 log 不会显示")
    custom_log.info("这是 INFO 日志")

    # 示例3: 多进程使用
    import multiprocessing
    import time

    def worker(worker_id):
        # 子进程中创建独立的日志器
        # 进程名会显示在日志格式的 %(processName)s 中
        worker_log = init_logger(f'worker_{worker_id}', 'logs')
        worker_log.info(f"工作进程 {worker_id} 启动")
        time.sleep(1)
        worker_log.info(f"工作进程 {worker_id} 完成")

    processes = []
    for i in range(3):
        p = multiprocessing.Process(target=worker, args=(i,), name=f"Worker-{i}")
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("所有进程完成！")
