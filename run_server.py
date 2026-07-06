"""
RAG 聊天系统 — 启动入口
=======================
使用 Uvicorn 启动 FastAPI 后端服务。

运行方式:
    python run_server.py

    或者直接使用命令行:
    uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

Uvicorn 参数说明:
    "server.main:app":
        - server.main: Python 模块路径（server 包下的 main.py）
        - app: 变量名（main.py 中的 app = FastAPI()）

    host (str, 默认 "0.0.0.0"):
        监听的主机地址。
        可选值:
            - "0.0.0.0": 监听所有网络接口（开发环境常用）
                优点: 可以本机访问，也可以局域网内其他设备访问
                缺点: 如果防火墙不配置，可能暴露给外部网络
            - "127.0.0.1" / "localhost": 仅本机可访问
                优点: 更安全，不会暴露给外部
                缺点: 局域网内其他设备无法访问
            - "192.168.x.x": 绑定到特定 IP 地址

    port (int, 默认 8000):
        监听端口。
        - 8000: 开发常用端口
        - 80: HTTP 标准端口（需要管理员权限）
        - 443: HTTPS 标准端口
        注意: 确保端口没有被其他程序占用

    reload (bool, 默认 True):
        是否开启热重载。
        - True: 代码文件发生变化时自动重启服务器（开发必开）
        - False: 不自动重启（生产环境应关闭，避免意外重启）
        原理: Uvicorn 监控 Python 文件的变化，检测到变化后自动重启进程
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",   # 应用路径: 模块.文件:变量
        host="0.0.0.0",      # 监听所有网络接口
        port=8000,           # 监听 8000 端口
        reload=True,         # 开发模式：代码变更时自动重启
    )
