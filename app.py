import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="centered")

try:
    from main import agent_executor
except Exception as e:
    st.error(f"Could not load the agent from main.py: {e}")
    st.stop()

EXAMPLES = [
    "Find the capital of India and its current weather",
    "What is the weather in Chennai right now?",
    "Who won the most recent Cricket World Cup?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []


def to_history(messages):
    history = []
    for m in messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))
    return history


def render_steps(steps):
    if not steps:
        return
    with st.expander(f"Tools used ({len(steps)})"):
        for step in steps:
            st.markdown(f"**{step['tool']}**")
            st.code(str(step["input"]), language="json")
            st.text(str(step["output"])[:1500])


with st.sidebar:
    st.header("Research Agent")
    st.caption("Groq · Tavily search · Weatherstack")

    st.subheader("Try an example")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.queued = example

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🔎 Research Agent")
st.caption("Ask anything. The agent searches the web and checks live weather when needed.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_steps(message.get("steps"))

typed = st.chat_input("Ask a question...")
user_input = st.session_state.pop("queued", None) or typed

if user_input:
    history = to_history(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": history,
                })
                answer = result["output"]
                steps = [
                    {
                        "tool": action.tool,
                        "input": action.tool_input,
                        "output": observation,
                    }
                    for action, observation in result.get("intermediate_steps", [])
                ]
            except Exception as e:
                answer = f"Something went wrong: {e}"
                steps = []

        st.markdown(answer)
        render_steps(steps)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "steps": steps}
    )