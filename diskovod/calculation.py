from __future__ import annotations

import ast
import math
import operator


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def evaluate_expression(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")
    count = 0

    def evaluate(node: ast.AST, depth: int = 0) -> int | float:
        nonlocal count
        count += 1
        if count > 50 or depth > 12:
            raise ValueError("expression is too complex")
        if isinstance(node, ast.Expression):
            return evaluate(node.body, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = evaluate(node.left, depth + 1)
            right = evaluate(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("exponent is too large")
            value = _BINARY_OPERATORS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            value = _UNARY_OPERATORS[type(node.op)](evaluate(node.operand, depth + 1))
        else:
            raise ValueError("unsupported expression")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 10**100:
            raise OverflowError("result is too large")
        return value

    return evaluate(tree)
