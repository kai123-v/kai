"""
自动生成 Google 风格 docstring（单函数 / 批量模式）
==================================================
用法:
    python gen_docstring.py <文件路径> <行号>
    python gen_docstring.py <文件路径> --all
"""
import ast
import sys
import re
from typing import Optional


def _type_to_str(node) -> str:
    if node is None: return "Any"
    if isinstance(node, ast.Name): return node.id
    if isinstance(node, ast.Subscript):
        base = _type_to_str(node.value)
        if isinstance(node.slice, ast.Tuple):
            args = ", ".join(_type_to_str(e) for e in node.slice.elts)
        else:
            args = _type_to_str(node.slice)
        return f"{base}[{args}]"
    if isinstance(node, ast.Constant) and node.value is None: return "None"
    if isinstance(node, ast.BinOp):
        return f"{_type_to_str(node.left)} | {_type_to_str(node.right)}"
    if isinstance(node, ast.Attribute): return node.attr
    return "Any"


def _extract_default(node) -> Optional[str]:
    if node is None: return None
    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, str): return f'"{val}"'
        if val is None: return "None"
        return str(val)
    if isinstance(node, ast.Name): return node.id
    return None


def _find_def_line(source_lines: list, target_line: int) -> Optional[int]:
    """从目标行向上找最近的 def 行，返回行号（1-based）"""
    for i in range(target_line - 1, -1, -1):
        stripped = source_lines[i].strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            return i + 1
        # 遇到空行或 class 或 decorator 继续
    return None


def _extract_signature(source_lines: list, def_line: int) -> Optional[str]:
    """提取完整的函数签名（可能跨多行），修复空体后返回可解析的代码片段"""
    lines = []
    # 收集从 def 行到包含 : 的行为止
    for i in range(def_line - 1, len(source_lines)):
        lines.append(source_lines[i])
        if ":" in source_lines[i]:
            break

    sig_text = "\n".join(lines)
    sig_text = sig_text.rstrip()

    # 计算基础缩进（def 行前面的空格数）
    def_raw = source_lines[def_line - 1]
    base_indent = len(def_raw) - len(def_raw.lstrip())

    # 去除基础缩进，让 ast.parse 能解析（顶层代码不能有缩进）
    dedented_lines = []
    for line in sig_text.split("\n"):
        if line.strip():  # 非空行
            if len(line) - len(line.lstrip()) >= base_indent:
                dedented_lines.append(line[base_indent:])
            else:
                dedented_lines.append(line.lstrip())
        else:
            dedented_lines.append(line)

    # 补一个 pass 体（4 空格缩进，因为已去除了基础缩进）
    if not dedented_lines[-1].strip().endswith("pass"):
        dedented_lines.append("    pass")

    return "\n".join(dedented_lines), base_indent


def _parse_single_function(source_lines: list, def_line: int):
    """从文件行列表中解析单个函数签名，返回 (ast_node, base_indent)"""
    result = _extract_signature(source_lines, def_line)
    if result is None:
        return None, 0
    sig, base_indent = result
    try:
        tree = ast.parse(sig)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node, base_indent
    except SyntaxError:
        pass
    return None, 0


def has_docstring(source_lines: list, def_line: int) -> bool:
    """检查 def 的下一行是否已有 docstring"""
    if def_line >= len(source_lines):
        return False
    next_line = source_lines[def_line].strip()  # def_line 是 1-based，下一行索引 = def_line
    return next_line.startswith('"""') or next_line.startswith("'''")


def _get_body_indent(source_lines: list, def_line: int) -> str:
    """获取函数体的缩进"""
    def_line_text = source_lines[def_line - 1]
    return " " * (len(def_line_text) - len(def_line_text.lstrip()) + 4)


def generate_docstring(node: ast.FunctionDef, indent: str = "    ") -> list:
    """为一个 AST 函数节点生成 Google 风格 docstring（返回行列表）"""
    params = []
    for arg in node.args.args:
        params.append({"name": arg.arg, "type": _type_to_str(arg.annotation), "default": None})
    defaults = node.args.defaults
    if defaults:
        offset = len(params) - len(defaults)
        for i, d in enumerate(defaults):
            params[offset + i]["default"] = _extract_default(d)
    if node.args.vararg:
        params.append({"name": f"*{node.args.vararg.arg}", "type": _type_to_str(node.args.vararg.annotation), "default": None})
    if node.args.kwarg:
        params.append({"name": f"**{node.args.kwarg.arg}", "type": _type_to_str(node.args.kwarg.annotation), "default": None})

    return_type = _type_to_str(node.returns) if node.returns else "None"

    lines = []
    lines.append(f'{indent}"""')
    lines.append(f"{indent}")
    if params:
        lines.append(f"{indent}Args:")
        for p in params:
            t = p["type"]
            d = p["default"]
            if d is not None:
                lines.append(f"{indent}    {p['name']} ({t}, optional): Defaults to {d}.")
            else:
                lines.append(f"{indent}    {p['name']} ({t}):")
    lines.append(f"{indent}")
    lines.append(f"{indent}Returns:")
    lines.append(f"{indent}    {return_type}:")
    lines.append(f'{indent}"""')
    return lines


def process_single(file_path: str, cursor_line: int) -> bool:
    """处理单个函数：在光标位置附近找到 def，生成 docstring 并插入"""
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    source_lines = source.split("\n")

    # 1. 找到最近的 def 行
    def_line = _find_def_line(source_lines, cursor_line)
    if def_line is None:
        print(f"  未找到函数定义", file=sys.stderr)
        return False

    # 2. 检查是否已有 docstring
    if has_docstring(source_lines, def_line):
        # 不做任何修改
        return True

    # 3. 单独解析这个函数签名
    func_node, base_indent = _parse_single_function(source_lines, def_line)
    if func_node is None:
        print(f"  无法解析函数签名 (第 {def_line} 行)", file=sys.stderr)
        return False

    # 4. 生成 docstring（使用正确的缩进）
    indent = " " * (base_indent + 4)
    doc_lines = generate_docstring(func_node, indent)

    # 5. 插入到 def 行之后
    new_lines = source_lines[:def_line] + doc_lines + source_lines[def_line:]
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))

    print(f"  + {func_node.name}")
    return True


def process_all(file_path: str) -> bool:
    """批量处理整个文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    source_lines = source.split("\n")

    # 尝试整体解析，失败则逐个处理
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # 文件中某个函数有空体 → 逐个 def 行单独解析
        tree = None

    if tree:
        nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        nodes.sort(key=lambda n: n.lineno, reverse=True)
        modified = 0
        for node in nodes:
            if has_docstring(source_lines, node.lineno):
                continue
            indent = " " * (node.col_offset + 4)
            doc_lines = generate_docstring(node, indent)
            source_lines = source_lines[:node.lineno] + doc_lines + source_lines[node.lineno:]
            modified += 1
            print(f"  + {node.name}")
    else:
        # 逐个 def 行处理
        modified = 0
        for i in range(len(source_lines)):
            stripped = source_lines[i].strip()
            if stripped.startswith("def ") or stripped.startswith("async def "):
                def_line = i + 1
                if has_docstring(source_lines, def_line):
                    continue
                func_node, base_indent = _parse_single_function(source_lines, def_line)
                if func_node is None:
                    continue
                indent = " " * (base_indent + 4)
                doc_lines = generate_docstring(func_node, indent)
                # 从后往前插入
                source_lines = source_lines[:def_line] + doc_lines + source_lines[def_line:]
                modified += 1
                print(f"  + {func_node.name}")

    if modified:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(source_lines))
        print(f"已更新: {file_path}")
    else:
        print("所有函数都已有 docstring。")
    return True


if __name__ == "__main__":
    if "--all" in sys.argv:
        process_all(sys.argv[1])
    else:
        file_path = sys.argv[1]
        line_number = int(sys.argv[2])
        process_single(file_path, line_number)
