"""
A minimal, restricted code-execution sandbox used by the Cleaner agent to
actually RUN the pandas code the LLM writes (instead of trusting it blindly),
capture stdout, and return any exception so the agent can retry.

This is intentionally simple (exec() with a locked-down builtins dict) --
good enough for a portfolio project running on data the user themself
uploaded. It is NOT a hardened sandbox for untrusted third-party code.
"""
import contextlib
import io
import traceback
from typing import Dict, Tuple, Optional

# Only expose the builtins that data-cleaning code plausibly needs.
_SAFE_BUILTINS = {
    "len": len, "range": range, "list": list, "dict": dict, "set": set,
    "str": str, "int": int, "float": float, "bool": bool, "tuple": tuple,
    "enumerate": enumerate, "zip": zip, "sorted": sorted, "min": min,
    "max": max, "sum": sum, "abs": abs, "round": round, "print": print,
    "isinstance": isinstance, "Exception": Exception, "ValueError": ValueError,
    "TypeError": TypeError, "KeyError": KeyError,
}


def extract_code(llm_text: str) -> str:
    """Strip ```python ... ``` fences (or plain ``` fences) from an LLM reply."""
    text = llm_text.strip()
    if "```" in text:
        parts = text.split("```")
        # parts alternate: [pre, code(+lang), post, ...] -> take first fenced block
        for i, p in enumerate(parts):
            if i % 2 == 1:  # inside a fence
                code = p
                if code.startswith("python"):
                    code = code[len("python"):]
                return code.strip()
    return text


def safe_exec(code: str, local_vars: Dict) -> Tuple[bool, str, Optional[str]]:
    """
    Executes `code` with `local_vars` as both locals and (extended) globals.
    Returns (success, stdout_text, error_text_or_None).
    """
    stdout_buffer = io.StringIO()
    exec_globals = dict(local_vars)
    exec_globals["__builtins__"] = _SAFE_BUILTINS
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, exec_globals, exec_globals)
        # copy results back so caller can pull out e.g. exec_globals['df']
        local_vars.update(
            {k: v for k, v in exec_globals.items() if k != "__builtins__"}
        )
        return True, stdout_buffer.getvalue(), None
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return False, stdout_buffer.getvalue(), f"{type(e).__name__}: {e}\n{tb}"
