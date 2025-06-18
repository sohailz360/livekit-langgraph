# LangGraph x LiveKit Agents

This is the Python implementation of LangGraph LiveKit Agents, which enables building voice-enabled AI agents using LangGraph and LiveKit.

## Initial Setup

1. Install the package as an editable dependency:

```bash
uv pip install -e .
```

2. Configure environment variables:
   Create a `.env` file with your API keys:

```
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
LIVEKIT_URL=
DEEPGRAM_API_KEY=
OPENAI_API_KEY=
GROQ_API_KEY=
```

3. Run the LangGraph dev server and the LiveKit Agents worker:

```bash
make start-agent
make start-voice
```

## Usage

```python
from langgraph_livekit_agents import LangGraphAdapter
from langgraph.pregel.remote import RemoteGraph

# Can either be a local compiled graph or a deployed graph
graph = RemoteGraph("agent", url="http://localhost:2024")

agent = pipeline.VoicePipelineAgent(
    vad=ctx.proc.userdata["vad"],
    stt=deepgram.STT(),
    llm=LangGraphAdapter(graph, config={"configurable": {"thread_id": thread_id}}),
    tts=openai.TTS(),
)
```

## Explanation

The LangGraph LiveKit adapter is a wrapper around `llm.LLM`, that maps the LangGraph `messages` stream mode to LiveKit's voice chunks.

The sample provides several key features:

1. **Interrupts**: Implement human-in-the-loop-style interrupts using LangGraph's `custom` stream mode. This allows the agent to pause and wait for user input at specific points in the conversation.

2. **Manual Voice Speakout**: Use the `say()` method to play static messages or announcements at any point in the conversation.

3. **Flushing**: The Python implementation includes a flushing mechanism to ensure all audio chunks are properly processed and played in sequence.

Here's an example of using these features:

```python
from langgraph.types import interrupt
from langgraph_livekit_agents.types import TypedLivekit

# Using interrupts to get user input
name, name_msgs = interrupt("What is your name?")

# Manual voice speakout
livekit = TypedLivekit(writer)
livekit.say("Give me a second to think...")

# Manually flush the TTS audio buffer
livekit.flush()
```

`LangGraphAdapter` supports both graphs deployed in LangGraph Platform and standalone graphs running within LiveKit Agents worker, just swap the `RemoteGraph` with your compiled graph.

```python
from langgraph_livekit_agents import LangGraphAdapter
from langgraph import StateGraph

# Can either be a local compiled graph or a deployed graph
graph = StateGraph(...)
    .compile()

agent = pipeline.VoicePipelineAgent(
    vad=ctx.proc.userdata["vad"],
    stt=deepgram.STT(),
    llm=LangGraphAdapter(graph),
    tts=openai.TTS(),
)
```
