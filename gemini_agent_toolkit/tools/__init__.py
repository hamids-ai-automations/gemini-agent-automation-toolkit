"""Tool declaration, registration and execution primitives."""

from .decorators import ToolSpec, build_input_model, get_tool_spec, tool
from .executor import ToolExecutor, make_function_response
from .registry import ToolRegistry

__all__ = [
    "ToolExecutor",
    "ToolRegistry",
    "ToolSpec",
    "build_input_model",
    "get_tool_spec",
    "make_function_response",
    "tool",
]
