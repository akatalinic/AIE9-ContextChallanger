from __future__ import annotations

import json
import logging
from typing import Any

from openai import BadRequestError

logger = logging.getLogger(__name__)

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text or ""))
except Exception:

    def count_tokens(text: str) -> int:
        return max(1, len(text or "") // 4)


def safe_json_parse(content: str, *, allow_array: bool = False) -> Any:
    raw = (content or "").strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    start_obj = raw.find("{")
    end_obj = raw.rfind("}")
    if start_obj >= 0 and end_obj > start_obj:
        try:
            return json.loads(raw[start_obj : end_obj + 1])
        except Exception:
            pass

    if allow_array:
        start_arr = raw.find("[")
        end_arr = raw.rfind("]")
        if start_arr >= 0 and end_arr > start_arr:
            try:
                return {"questions": json.loads(raw[start_arr : end_arr + 1])}
            except Exception:
                pass

    return {}


def create_chat_completion_with_fallback(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_output_tokens: int,
    timeout: float | None = None,
    response_format: dict[str, str] | None = None,
    seed: int | None = None,
    log_label: str = "LLM",
) -> Any:
    common_kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
    }
    if timeout is not None:
        common_kwargs["timeout"] = timeout
    if response_format is not None:
        common_kwargs["response_format"] = response_format
    if seed is not None:
        common_kwargs["seed"] = seed

    token_param = "max_tokens"
    while True:
        try:
            request_kwargs = dict(common_kwargs)
            request_kwargs[token_param] = max_output_tokens
            return client.chat.completions.create(**request_kwargs)
        except BadRequestError as exc:
            message = str(exc).lower()
            if "seed" in common_kwargs and "seed" in message and "unsupported" in message:
                logger.info("%s seed fallback applied | model=%s removing_seed", log_label, model)
                common_kwargs.pop("seed", None)
                continue
            if (
                token_param == "max_tokens"
                and "max_tokens" in message
                and "max_completion_tokens" in message
            ):
                logger.info(
                    "%s token parameter fallback applied | model=%s using=max_completion_tokens",
                    log_label,
                    model,
                )
                token_param = "max_completion_tokens"
                continue
            raise
