"""
FastAPI 应用工厂模块
=====================
负责创建和配置 FastAPI 应用实例，包括 CORS 跨域、路由注册、健康检查。
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.api import router

# 创建 FastAPI 应用实例
# title: API 文档标题，会显示在 Swagger UI 和 ReDoc 中
# version: API 版本号，用于文档标注
app = FastAPI(title="RAG Chat API", version="1.0.0")

# ==================== CORS 跨域配置 ====================
# 允许前端开发服务器（如 Vite 的 localhost:5173）访问后端 API
# 如果没有此配置，浏览器会因同源策略阻止跨域请求
app.add_middleware(
    CORSMiddleware,
    # allow_origins: 允许哪些域名访问
    #   - ["*"] 表示允许所有域名（开发环境使用，生产环境应限制为具体域名）
    #   - 也可以指定具体域名列表，如 ["http://localhost:5173", "https://你的域名.com"]
    allow_origins=["*"],
    # allow_credentials: 是否允许携带 Cookie 和认证信息（如 Authorization header）
    #   - True: 允许跨域请求携带凭据
    #   - 注意：allow_origins=["*"] 时，allow_credentials 必须为 True（否则某些浏览器会拒绝）
    allow_credentials=True,
    # allow_methods: 允许的 HTTP 方法
    #   - ["*"] 允许所有方法（GET, POST, PUT, DELETE, PATCH, OPTIONS 等）
    #   - 也可以指定为 ["GET", "POST"] 等具体方法列表
    allow_methods=["*"],
    # allow_headers: 允许的请求头
    #   - ["*"] 允许所有请求头
    #   - 也可以指定为 ["Content-Type", "Authorization"] 等具体头部列表
    allow_headers=["*"],
)

# ==================== 路由注册 ====================
# 将 api.py 中定义的路由注册到应用
# prefix="/api": 所有路由前添加 /api 前缀
#   例如 api.py 中定义 @router.post("/chat")
#   实际访问路径为 POST /api/chat
app.include_router(router, prefix="/api")


# ==================== 健康检查端点 ====================
@app.get("/health")
async def health():
    """
    健康检查接口
    =============
    用途：供监控系统、负载均衡器或 Docker 健康检查调用
    返回 200 OK 表示服务正常运行

    Returns:
        dict: {"status": "ok"} — 服务正常
    """
    return {"status": "ok"}
