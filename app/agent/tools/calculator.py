"""Safe arithmetic evaluator — AST walk instead of ``eval``.

The agent is allowed to compute, never to execute. Every node type is
whitelisted explicitly, exponent size and input length are bounded, so a
hostile prompt cannot reach the interpreter.
"""

from __future__ import annotations

import ast
import math
import operator

from langchain_core.tools import tool

_MAX_INPUT_LENGTH = 200
_MAX_EXPONENT = 64

_BIN_OPS: dict[type[ast.operator], object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], object] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, object] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("只支持数字常量")
        # Keep int/float as written: round(pi, 4) needs an integer ndigits.
        return node.value

    if isinstance(node, ast.BinOp):
        func = _BIN_OPS.get(type(node.op))
        if func is None:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("指数过大")
        return func(left, right)  # type: ignore[operator]

    if isinstance(node, ast.UnaryOp):
        func = _UNARY_OPS.get(type(node.op))
        if func is None:
            raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
        return func(_evaluate(node.operand))  # type: ignore[operator]

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"未知常量: {node.id}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ValueError("只支持白名单内的数学函数")
        if node.keywords:
            raise ValueError("不支持关键字参数")
        args = [_evaluate(arg) for arg in node.args]
        return _FUNCTIONS[node.func.id](*args)  # type: ignore[operator]

    raise ValueError(f"不支持的表达式片段: {type(node).__name__}")


def _format(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("结果不是一个有限数字")
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    # repr() gives the shortest string that round-trips, so no precision is lost
    # to formatting (0.1 + 0.2 stays 0.30000000000000004 instead of 0.3).
    return repr(value)


@tool
def calculator(expression: str) -> str:
    """Evaluate a single arithmetic expression and return the exact result.

    Supports ``+ - * / // % **``, parentheses, and a small math library
    (``sqrt``, ``log``, ``exp``, ``sin``, ``round``, ``abs``, ``min``, ``max``,
    ...) plus the constants ``pi``, ``e``, ``tau``.

    Use this instead of doing mental arithmetic whenever numbers are involved.

    Args:
        expression: e.g. ``"(1200 * 0.85) ** 2 / 7"`` or ``"sqrt(144) + 3"``.
    """
    text = (expression or "").strip()
    if not text:
        return "错误: 表达式为空"
    if len(text) > _MAX_INPUT_LENGTH:
        return f"错误: 表达式超过 {_MAX_INPUT_LENGTH} 字符"
    try:
        result = _evaluate(ast.parse(text, mode="eval"))
    except ZeroDivisionError:
        return "错误: 除数为零"
    except (SyntaxError, ValueError, TypeError, OverflowError) as exc:
        return f"错误: {exc}"
    except Exception as exc:
        return f"错误: 无法计算（{type(exc).__name__}）"
    try:
        return _format(result)
    except ValueError as exc:
        return f"错误: {exc}"
