"""Azure OpenAI wrapper: chat, validated JSON output, and a tool-calling loop.

The tool loop is written out by hand (no agent framework) to keep the
protocol visible: the model returns tool_calls, we run them (in parallel),
send each result back as a "tool" message, and repeat until the model
answers in text or the step budget runs out.
"""
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

from openai import AzureOpenAI
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class ToolOutput:
    """What a tool returns: text for the model, sources for the registry."""
    text: str
    sources: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Tool:
    """A function the model can call, with a Pydantic argument schema."""
    name: str
    description: str
    args: Type[BaseModel]
    fn: Callable[[Any], ToolOutput]

    def schema(self) -> Dict[str, Any]:
        """OpenAI function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args.model_json_schema(),
            },
        }

    def invoke(self, raw_args: str) -> ToolOutput:
        return self.fn(self.args.model_validate_json(raw_args or "{}"))


class LLM:
    """Thin client with token accounting shared across threads."""

    def __init__(self, endpoint: str, key: str, deployment: str, api_version: str):
        self.client = AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=api_version)
        self.deployment = deployment
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        self._lock = threading.Lock()

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        json_mode: bool = False,
        max_tokens: int = 1500,
        temperature: float = 0.2,
    ):
        """One chat completion; returns the assistant message."""
        kwargs: Dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        with self._lock:
            self.usage["calls"] += 1
            if response.usage:
                self.usage["prompt_tokens"] += response.usage.prompt_tokens
                self.usage["completion_tokens"] += response.usage.completion_tokens
        return response.choices[0].message

    def structured(
        self,
        messages: List[Dict[str, Any]],
        model: Type[T],
        max_tokens: int = 2500,
        retries: int = 1,
    ) -> T:
        """
        JSON-mode completion validated against a Pydantic model. On invalid
        output, the validation error is sent back once so the model can fix it.
        """
        messages = list(messages)
        for attempt in range(retries + 1):
            message = self.chat(messages, json_mode=True, max_tokens=max_tokens)
            try:
                return model.model_validate_json(message.content or "")
            except ValidationError as e:
                if attempt == retries:
                    raise
                logger.warning(f"Invalid {model.__name__} output, retrying: {e}")
                messages += [
                    {"role": "assistant", "content": message.content or ""},
                    {"role": "user", "content": f"That JSON failed validation:\n{e}\nReturn corrected JSON only."},
                ]
        raise AssertionError("unreachable")

    def run_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Dict[str, Tool],
        max_steps: int = 5,
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Tool-calling loop.

        Tool errors (bad arguments, exceptions) go back to the model as the
        tool result, so it can correct itself instead of crashing the run.

        Returns:
            (final answer, sources collected from tool results, trace of calls)
        """
        messages = list(messages)
        schemas = [tool.schema() for tool in tools.values()]
        sources: List[Dict[str, Any]] = []
        trace: List[Dict[str, Any]] = []

        for _ in range(max_steps):
            message = self.chat(messages, tools=schemas)
            if not message.tool_calls:
                return message.content or "", sources, trace

            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [call.model_dump() for call in message.tool_calls],
            })
            # The model may request several calls in one turn; run them concurrently
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda call: self._run_call(call, tools), message.tool_calls))
            for call, (output, entry) in zip(message.tool_calls, results):
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output.text})
                sources.extend(output.sources)
                trace.append(entry)

        # Budget spent: one last call with no tools forces an answer
        messages.append({"role": "user", "content": "Tool budget reached. Answer now with what you have."})
        return self.chat(messages).content or "", sources, trace

    @staticmethod
    def _run_call(call, tools: Dict[str, Tool]) -> Tuple[ToolOutput, Dict[str, Any]]:
        start = time.perf_counter()
        name, raw_args = call.function.name, call.function.arguments
        tool = tools.get(name)
        try:
            if tool is None:
                raise ValueError(f"Unknown tool {name!r}. Available: {sorted(tools)}")
            output = tool.invoke(raw_args)
            error = None
        except Exception as e:  # any failure becomes an observation for the model
            output = ToolOutput(text=f"ERROR: {type(e).__name__}: {e}")
            error = str(e)
        entry = {
            "tool": name,
            "args": _safe_json(raw_args),
            "ms": round((time.perf_counter() - start) * 1000),
            "sources": len(output.sources),
            "error": error,
            "preview": output.text[:200],
        }
        return output, entry


def _safe_json(raw: str) -> Any:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return raw
