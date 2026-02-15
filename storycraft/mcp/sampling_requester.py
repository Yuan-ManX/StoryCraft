from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession
from mcp.types import (
    SamplingMessage,
    TextContent,
    ModelHint,
    ModelPreferences,
)

from storycraft.utils.emoji import EmojiManager


# ============================================================
# Protocols
# ============================================================

class BaseLLMSampling(Protocol):
    """
    Low-level protocol for sampling requests.

    This interface represents the *transport layer*, which may be:
    - MCP server
    - HTTP REST API
    - WebSocket streaming
    - Local inference runtime
    """

    async def sampling(
        self,
        *,
        system_prompt: str | None,
        messages: list[SamplingMessage],
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        model_preferences: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        stop_sequences: list[str] | None = None,
    ) -> str:
        ...


@runtime_checkable
class LLMClient(Protocol):
    """
    High-level unified LLM client interface.

    Tools only care about:
        text  vs  multimodal
    """

    async def complete(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        media: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        model_preferences: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        stop_sequences: list[str] | None = None,
    ) -> str:
        ...


# ============================================================
# MCP Sampler Adapter
# ============================================================

class MCPSampler(BaseLLMSampling):
    """
    MCP adapter: convert local sampling calls into MCP protocol requests.
    """

    def __init__(self, mcp_ctx: Context[ServerSession, object]):
        self._ctx = mcp_ctx
        self._emoji = EmojiManager()

    # -------------------------
    # Internal helpers
    # -------------------------

    def _to_model_preferences(
        self,
        model_preferences: dict[str, Any] | None,
    ) -> Optional[ModelPreferences]:
        """
        Convert dict-style preferences into MCP-native ModelPreferences.
        """
        if not model_preferences:
            return None

        raw_hints = model_preferences.get("hints")
        hints: list[ModelHint] | None = None

        if isinstance(raw_hints, list):
            hints = []
            for h in raw_hints:
                if isinstance(h, ModelHint):
                    hints.append(h)
                elif isinstance(h, dict):
                    hints.append(ModelHint(**h))
                elif isinstance(h, str):
                    hints.append(ModelHint(name=h))

        return ModelPreferences(
            hints=hints,
            costPriority=model_preferences.get("costPriority"),
            speedPriority=model_preferences.get("speedPriority"),
            intelligencePriority=model_preferences.get("intelligencePriority"),
        )

    def _extract_text(self, content: Any) -> str:
        """
        Extract plain text from MCP multimodal blocks.
        """
        if isinstance(content, list):
            texts = [
                block.text
                for block in content
                if getattr(block, "type", None) == "text"
            ]
            return self._emoji.remove_emoji("\n".join(texts).strip())

        if getattr(content, "type", None) == "text":
            return self._emoji.remove_emoji(content.text.strip())

        return self._emoji.remove_emoji(str(content))

    def _merge_metadata(
        self,
        metadata: dict[str, Any] | None,
        *,
        top_p: float,
    ) -> dict[str, Any]:
        """
        Merge runtime metadata with sampling configuration.
        """
        merged = dict(metadata or {})
        merged["top_p"] = top_p
        return merged

    # -------------------------
    # Core sampling
    # -------------------------

    async def sampling(
        self,
        *,
        system_prompt: str | None,
        messages: list[SamplingMessage],
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        model_preferences: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        stop_sequences: list[str] | None = None,
    ) -> str:
        merged_metadata = self._merge_metadata(metadata, top_p=top_p)

        result = await self._ctx.session.create_message(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=merged_metadata,
            # model_preferences=self._to_model_preferences(model_preferences),
            # stop_sequences=stop_sequences,
        )

        return self._extract_text(result.content)


# ============================================================
# High-level LLM Client
# ============================================================

class SamplingLLMClient(LLMClient):
    """
    Unified LLM client built on BaseLLMSampling.

    Responsibility:
        - Normalize input
        - Assemble MCP messages
        - Handle multimodal routing
    """

    def __init__(self, sampler: BaseLLMSampling):
        self._sampler = sampler

    def _build_messages(self, user_prompt: str) -> list[SamplingMessage]:
        return [
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text=user_prompt),
            )
        ]

    def _merge_metadata(
        self,
        metadata: dict[str, Any] | None,
        *,
        media: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        merged = dict(metadata or {})
        merged["modality"] = "multimodal" if media else "text"

        if media:
            # Critical: propagate raw paths & timestamps
            merged["media"] = media

        return merged

    async def complete(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        media: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        model_preferences: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        stop_sequences: list[str] | None = None,
    ) -> str:
        messages = self._build_messages(user_prompt)

        merged_metadata = self._merge_metadata(
            metadata,
            media=media,
        )

        return await self._sampler.sampling(
            system_prompt=system_prompt,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            model_preferences=model_preferences,
            metadata=merged_metadata,
            stop_sequences=stop_sequences,
        )


# ============================================================
# Factory
# ============================================================

def make_llm(mcp_ctx: Context[ServerSession, object]) -> LLMClient:
    """
    Factory for tool-level LLM client.
    """
    return SamplingLLMClient(MCPSampler(mcp_ctx))

