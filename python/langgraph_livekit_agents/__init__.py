"""
LangGraphAdapter masquerades as an livekit.LLM and translates the LiveKit chat chunks
into LangGraph messages.
"""

from typing import Any, Optional, Dict
from livekit.agents import llm
from langgraph.pregel import PregelProtocol
from langchain_core.messages import BaseMessageChunk, AIMessage, HumanMessage
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.tts import SynthesizeStream
from livekit.agents.utils import shortuuid
from langgraph.types import Command
from langgraph.errors import GraphInterrupt
from httpx import HTTPStatusError

import logging

logger = logging.getLogger(__name__)


# https://github.com/livekit/agents/issues/1370#issuecomment-2588821571
class FlushSentinel(str, SynthesizeStream._FlushSentinel):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)


class LangGraphStream(llm.LLMStream):
    def __init__(
        self,
        llm: llm.LLM,
        chat_ctx: llm.ChatContext,
        graph: PregelProtocol,
        tools: list[llm.FunctionTool | llm.RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = None,
    ):
        super().__init__(
            llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options
        )
        self._graph = graph

    async def _run(self):
        # Change 1) Instead of converting all messages, we just now take the last human message
        input_human_message = next(
            (
                self._to_message(m)
                for m in reversed(self.chat_ctx.items)
                if m.role == "user"
            ),
            None,
        )

        messages = [input_human_message] if input_human_message else []
        input = {"messages": messages}

        # see if we need to respond to an interrupt
        if interrupt := await self._get_interrupt():
            used_messages = [
                AIMessage(interrupt.value),
                input_human_message,
            ]

            input = Command(resume=(input_human_message.content, used_messages))

        try:
            async for mode, data in self._graph.astream(
                input, config=self._llm._config, stream_mode=["messages", "custom"]
            ):
                if mode == "messages":
                    if chunk := await self._to_livekit_chunk(data[0]):
                        self._event_ch.send_nowait(chunk)

                if mode == "custom":
                    if isinstance(data, dict) and (event := data.get("type")):
                        if event == "say" or event == "flush":
                            content = (data.get("data") or {}).get("content")
                            if chunk := await self._to_livekit_chunk(content):
                                self._event_ch.send_nowait(chunk)
                            print("SAY METHOD GOT CALLED SHOPULD SPEAK IMMEDIETLY")
                            self._event_ch.send_nowait(
                                self._create_livekit_chunk(FlushSentinel())
                            )
        except GraphInterrupt:
            pass

        # If interrupted, send the string as a message
        if interrupt := await self._get_interrupt():
            if chunk := await self._to_livekit_chunk(interrupt.value):
                self._event_ch.send_nowait(chunk)

    async def _get_interrupt(cls) -> Optional[str]:
        try:
            state = await cls._graph.aget_state(config=cls._llm._config)
            interrupts = [
                interrupt for task in state.tasks for interrupt in task.interrupts
            ]
            assistant = next(
                (
                    interrupt
                    for interrupt in reversed(interrupts)
                    if isinstance(interrupt.value, str)
                ),
                None,
            )
            return assistant
        except HTTPStatusError as e:
            return None

    def _to_message(cls, msg: llm.ChatMessage) -> HumanMessage:
        if isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, list):
            content = []
            for c in msg.content:
                if isinstance(c, str):
                    content.append({"type": "text", "text": c})
                elif isinstance(c, llm.ChatImage):
                    if isinstance(c.image, str):
                        content.append({"type": "image_url", "image_url": c.image})
                    else:
                        logger.warning("Unsupported image type")
                else:
                    logger.warning("Unsupported content type")
        else:
            content = ""

        return HumanMessage(content=content, id=msg.id)

    @staticmethod
    def _create_livekit_chunk(
        content: str,
        *,
        id: str | None = None,
    ) -> llm.ChatChunk | None:
        return llm.ChatChunk(
            id=id or shortuuid(),
            
            delta=llm.ChoiceDelta(role="assistant", content=content)
            
        )

    @staticmethod
    async def _to_livekit_chunk(
        msg: BaseMessageChunk | str | None,
    ) -> llm.ChatChunk | None:
        if not msg:
            return None

        request_id = None
        content = msg

        if isinstance(msg, str):
            content = msg
        elif hasattr(msg, "content") and isinstance(msg.content, str):
            request_id = getattr(msg, "id", None)
            content = msg.content
        elif isinstance(msg, dict):
            request_id = msg.get("id")
            content = msg.get("content")

        return LangGraphStream._create_livekit_chunk(content, id=request_id)


class LangGraphAdapter(llm.LLM):
    def __init__(self, graph: Any, config: dict[str, Any] | None = None):
        super().__init__()
        self._graph = graph
        self._config = config

    def chat(
        self,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        tools=None,
        **kwargs,
    ) -> llm.LLMStream:
        return LangGraphStream(
            self,
            chat_ctx=chat_ctx,
            graph=self._graph,
            tools=tools,
            conn_options=conn_options,
        )
