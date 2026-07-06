"""
用本地浏览器渲染 Mermaid 图为 PNG。

用法:
    python render_graphs.py
"""

import sys, os, base64
sys.path.insert(0, ".")

os.chdir(os.path.dirname(os.path.abspath(__file__)))


GRAPH1_MMD = """graph TD;
    __start__([__start__])
    agent(agent)
    retrieve(retrieve)
    rewrite(rewrite)
    generate(generate)
    __end__([__end__])
    __start__ --> agent;
    agent -. &nbsp;tools&nbsp; .-> retrieve;
    agent -.-> __end__;
    retrieve -. &nbsp;相关&nbsp; .-> generate;
    retrieve -. &nbsp;不相关&nbsp; .-> rewrite;
    rewrite --> agent;
    generate --> __end__;"""

GRAPH2_MMD = """graph TD;
    __start__([__start__])
    web_search(web_search)
    retrieve(retrieve)
    grade_documents(grade_documents)
    generate(generate)
    transformer_query(transformer_query)
    __end__([__end__])
    __start__ -. &nbsp;vectorstore&nbsp; .-> retrieve;
    __start__ -. &nbsp;web_search&nbsp; .-> web_search;
    retrieve --> grade_documents;
    grade_documents -. &nbsp;有相关文档&nbsp; .-> generate;
    grade_documents -. &nbsp;无文档+次数<2&nbsp; .-> transformer_query;
    grade_documents -. &nbsp;无文档+次数≥2&nbsp; .-> web_search;
    transformer_query --> retrieve;
    web_search --> generate;
    generate -. &nbsp;useful&nbsp; .-> __end__;
    generate -. &nbsp;not supported&nbsp; .-> generate;
    generate -. &nbsp;not useful&nbsp; .-> transformer_query;"""

TITLE_MAP = {
    "graph1": "Graph1 — 基础 RAG (Agent + 检索评估)",
    "graph2": "Graph2 — Corrective RAG (路由 + 幻觉检测)",
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad:true, theme:'default', flowchart:{{curve:'linear'}}}});</script>
<style>body{{margin:0;padding:24px;background:#fff;font-family:Arial,sans-serif}}
.title{{text-align:center;font-size:16px;font-weight:700;margin-bottom:16px;color:#333}}
.mermaid{{display:flex;justify-content:center}}</style></head>
<body>
<div class="title">{title}</div>
<div class="mermaid">{mmd}</div>
</body></html>"""


def render(mmd: str, title: str, output: str):
    """用 Playwright 渲染 Mermaid 图并截图。"""
    html = HTML_TEMPLATE.format(title=title, mmd=mmd)

    with open("_temp.html", "w", encoding="utf-8") as f:
        f.write(html)

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            headless=True,
        )
        page = browser.new_page(viewport={"width": 900, "height": 600})
        page.goto("file:///" + os.path.abspath("_temp.html").replace("\\", "/"))
        # 等 Mermaid 渲染完成
        page.wait_for_selector("svg", timeout=15000)
        page.wait_for_timeout(1000)  # 额外等动画完成
        # 截图
        el = page.query_selector(".mermaid")
        el.screenshot(path=output)
        browser.close()

    os.remove("_temp.html")
    print(f"{output} 已生成")


if __name__ == "__main__":
    render(GRAPH1_MMD, TITLE_MAP["graph1"], "rag_agent_graph.png")
    render(GRAPH2_MMD, TITLE_MAP["graph2"], "graph2_crag.png")
