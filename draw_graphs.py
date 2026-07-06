"""
自动生成 Graph1 和 Graph2 的 Mermaid 流程图。

用法:
    python draw_graphs.py           # 生成 .mmd 文件
    python draw_graphs.py --png     # 生成 .png 图片（需要安装 mmdc 命令）
"""

import sys
sys.path.insert(0, ".")

from rag.graph.graph1 import graph as graph1
from rag.graph2.graph_2 import graph as graph2

GRAPH1_MMD = """---
config:
  flowchart:
    curve: linear
---
graph TD;
    __start__([<p>__start__</p>]):::first
    agent(agent)
    retrieve(retrieve)
    rewrite(rewrite)
    generate(generate)
    __end__([<p>__end__</p>]):::last
    __start__ --> agent;
    agent -. &nbsp;tools&nbsp; .-> retrieve;
    agent -.-> __end__;
    retrieve -. &nbsp;相关&nbsp; .-> generate;
    retrieve -. &nbsp;不相关&nbsp; .-> rewrite;
    rewrite --> agent;
    generate --> __end__;
    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
"""

GRAPH2_MMD = """---
config:
  flowchart:
    curve: linear
---
graph TD;
    __start__([<p>__start__</p>]):::first
    web_search(web_search)
    retrieve(retrieve)
    grade_documents(grade_documents)
    generate(generate)
    transformer_query(transformer_query)
    __end__([<p>__end__</p>]):::last
    __start__ -. &nbsp;vectorstore&nbsp; .-> retrieve;
    __start__ -. &nbsp;web_search&nbsp; .-> web_search;
    retrieve --> grade_documents;
    grade_documents -. &nbsp;有相关文档&nbsp; .-> generate;
    grade_documents -. &nbsp;无文档+次数&lt;2&nbsp; .-> transformer_query;
    grade_documents -. &nbsp;无文档+次数≥2&nbsp; .-> web_search;
    transformer_query --> retrieve;
    web_search --> generate;
    generate -. &nbsp;useful&nbsp; .-> __end__;
    generate -. &nbsp;not supported&nbsp; .-> generate;
    generate -. &nbsp;not useful&nbsp; .-> transformer_query;
    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
"""


def write_mmd_files():
    with open("rag_agent_graph.mmd", "w", encoding="utf-8") as f:
        f.write(GRAPH1_MMD)
    print("rag_agent_graph.mmd (Graph1) 已生成")

    with open("graph2_crag.mmd", "w", encoding="utf-8") as f:
        f.write(GRAPH2_MMD)
    print("graph2_crag.mmd (Graph2) 已生成")


def write_png_files():
    import subprocess, os
    for mmd, png in [("rag_agent_graph.mmd", "rag_agent_graph.png"),
                     ("graph2_crag.mmd", "graph2_crag.png")]:
        if os.path.exists(mmd):
            subprocess.run(
                ["mmdc", "-i", mmd, "-o", png, "-b", "transparent"],
                check=True,
            )
            print(f"{png} 已生成")
        else:
            print(f"跳过: {mmd} 不存在，请先运行不带 --png 的命令生成 .mmd 文件")


if __name__ == "__main__":
    write_mmd_files()
    if "--png" in sys.argv:
        write_png_files()
