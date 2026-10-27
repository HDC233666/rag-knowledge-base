"""调用 DeepSeek 大模型（OpenAI 兼容接口），带重试和友好报错。"""

from __future__ import annotations

import time

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, TEMPERATURE


class LLMError(RuntimeError):
    """对用户友好的大模型调用错误（API 层会转成 502）。"""


def get_client() -> OpenAI:
    if not DEEPSEEK_API_KEY:
        raise LLMError(
            "没有找到 DEEPSEEK_API_KEY。"
            "请在项目根目录的 .env 里填写，或设置同名的系统环境变量后重启。"
        )
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def chat(messages: list[dict], temperature: float | None = None) -> dict:
    """发一次对话请求，返回 {text, latency_ms, tokens}。"""
    client = get_client()
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            started = time.time()
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                temperature=TEMPERATURE if temperature is None else temperature,
            )
            text = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            return {
                "text": text,
                "latency_ms": int((time.time() - started) * 1000),
                "tokens": getattr(usage, "total_tokens", None) if usage else None,
            }
        except (APIConnectionError, APITimeoutError) as exc:  # 网络抖动：重试
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
        except APIError as exc:  # 鉴权 / 模型名 / 限流：直接报错
            raise LLMError(f"DeepSeek 接口报错：{exc}") from exc

    raise LLMError(f"网络重试 3 次仍然失败：{last_error}")
