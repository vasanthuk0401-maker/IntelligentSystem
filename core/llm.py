"""Thin wrapper around the Claude API used by every agent.

Every agent asks for *structured* output (a Pydantic model) via
`client.messages.parse`. If no credentials are configured, or a call fails,
`parse()` returns None and the agent falls back to its offline heuristic, so the
whole pipeline still runs end-to-end without an API key.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from core import config

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _has_credentials() -> bool:
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return True
    # `ant auth login` stores a profile the SDK picks up automatically.
    return (Path.home() / ".config" / "anthropic").exists()


class LLM:
    def __init__(self, model: str | None = None, enabled: bool | None = None):
        self.model = model or config.CLAUDE_MODEL
        self.enabled = _has_credentials() if enabled is None else enabled
        self.last_error: str = ""
        self._client = None
        self._use_fallbacks = True
        if self.enabled:
            try:
                import anthropic

                self._anthropic = anthropic
                self._client = anthropic.Anthropic(max_retries=3)
            except Exception as e:  # SDK missing or misconfigured
                self.enabled = False
                self.last_error = f"Claude client unavailable: {e}"

    @property
    def mode(self) -> str:
        return f"Claude ({self.model})" if self.enabled else "Offline heuristics"

    def parse(
        self,
        system: str,
        user: str,
        schema: Type[T],
        effort: str = "low",
        max_tokens: int = 16000,
    ) -> Optional[T]:
        """Return a validated `schema` instance, or None if Claude is unavailable."""
        if not self.enabled or self._client is None:
            return None
        anthropic = self._anthropic
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config={"effort": effort},
        )
        try:
            if self._use_fallbacks:
                try:
                    # Server-side refusal fallback: if a safety classifier declines,
                    # the API re-runs the request on a fallback model in the same call.
                    resp = self._client.beta.messages.parse(
                        betas=[FALLBACK_BETA], fallbacks="default", **kwargs
                    )
                except (TypeError, anthropic.BadRequestError) as e:
                    log.info("Fallbacks unsupported here, using plain parse: %s", e)
                    self._use_fallbacks = False
                    resp = self._client.messages.parse(**kwargs)
            else:
                resp = self._client.messages.parse(**kwargs)

            if resp.stop_reason == "refusal":
                self.last_error = "Claude declined this request."
                return None
            if resp.stop_reason == "max_tokens":
                self.last_error = "Response truncated (max_tokens)."
                return None
            return resp.parsed_output
        except anthropic.AuthenticationError:
            self.last_error = "Invalid API key - switched to offline heuristics."
            self.enabled = False
        except anthropic.RateLimitError:
            self.last_error = "Rate limited by Claude API - used heuristics for this step."
        except anthropic.APIConnectionError:
            self.last_error = "Network error reaching Claude - used heuristics for this step."
        except anthropic.APIStatusError as e:
            self.last_error = f"Claude API error {e.status_code}: {e.message}"
        except Exception as e:  # validation errors etc.
            self.last_error = f"Unexpected LLM error: {e}"
        log.warning(self.last_error)
        return None


_default: Optional[LLM] = None


def get_llm() -> LLM:
    global _default
    if _default is None:
        _default = LLM()
    return _default
