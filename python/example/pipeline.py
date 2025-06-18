import logging
from uuid import uuid4, uuid5, UUID
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    Agent,
    AgentSession
    
)
from livekit.plugins import  deepgram, silero
from langgraph_livekit_agents import LangGraphAdapter
from langgraph.pregel.remote import RemoteGraph

load_dotenv(dotenv_path=".env")
logger = logging.getLogger("voice-agent")


def get_thread_id(sid: str | None) -> str:
    NAMESPACE = UUID("41010b5d-5447-4df5-baf2-97d69f2e9d06")
    if sid is not None:
        return str(uuid5(NAMESPACE, sid))
    return str(uuid4())


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


class Assistant(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful AI assistant.Always provide summarized and short answers.Be super chatty .",
)
async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    thread_id = get_thread_id(participant.sid)

    logger.info(
        f"starting voice assistant for participant {participant.identity} (thread ID: {thread_id})"
    )
    graph = RemoteGraph("agent", url="http://localhost:2024")
    llm_adapter = LangGraphAdapter(
        graph=graph,
        config={"configurable": {"thread_id": thread_id}},
        #AudioPlayerLivekit=background_audio,
        #langgraph_node=["say_filler","call_model"]
        # note: we *don’t* yet pass session here
    )
    
    agent = Agent(
    instructions="You are a helpful AI assistant.Always provide summarized and short answers.Be super chatty .",
    
        )

    
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
       llm=llm_adapter,
        tts=deepgram.TTS(),
    )

    await session.start(room=ctx.room, agent=Assistant())


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
