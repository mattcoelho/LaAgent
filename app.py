import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage

# 1. PAGE CONFIG
st.set_page_config(page_title="Bridge of Death", layout="wide")
st.title("🧌 The Bridge of Death (Secrets Edition)")

# 2. SECURE API KEY RETRIEVAL (The "Production" Setup)
# This looks for GROQ_API_KEY in .streamlit/secrets.toml (Local) or Streamlit Cloud Secrets
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("🚨 Missing API Key! Please add `GROQ_API_KEY` to your secrets.")
    st.stop()

# 3. STATE MANAGEMENT (The "Bridge" Logic)
if "troll_stage" not in st.session_state:
    st.session_state.troll_stage = 0  # 0: Name, 1: Quest, 2: Color, 3: PASSED
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "STOP! Who would cross the Bridge of Death must answer me these questions three, ere the other side he see."})

# 4. DEFINE TOOLS (The "Governance")
@tool
def submit_answer(answer_is_acceptable: bool):
    """
    Call this tool when the user provides a valid answer to the current question.
    If the answer is gibberish, do NOT call this.
    Returns a status message that will be used to update the stage.
    """
    if answer_is_acceptable:
        return "STATE_UPDATE: ADVANCE_STAGE"
    return "Answer rejected."

@tool
def cast_into_gorge():
    """Call this if the user answers the 'Color' question incorrectly or acts rude."""
    return "STATE_UPDATE: RESET_BRIDGE"

tools = [submit_answer, cast_into_gorge]

# 5. DYNAMIC SYSTEM PROMPT (The "Persona" Logic)
current_stage = st.session_state.troll_stage

if current_stage == 0:
    system_instruction = (
        "You are the Keeper of the Bridge of Death. "
        "The user MUST tell you their NAME. "
        "If they give you a name, use the submit_answer tool with answer_is_acceptable=True. "
        "If they ask anything else, yell 'STOP!' and demand their name again. "
        "IMPORTANT: Use tools through the system's tool calling mechanism. Do NOT write tool calls as text or XML."
    )
    current_question = "What... is your name?"

elif current_stage == 1:
    system_instruction = (
        "You are the Keeper of the Bridge of Death. "
        "The user has given their name. Now they MUST tell you their QUEST. "
        "If they state a quest, use the submit_answer tool with answer_is_acceptable=True. "
        "Do not chat. Just demand the quest. "
        "IMPORTANT: Use tools through the system's tool calling mechanism. Do NOT write tool calls as text or XML."
    )
    current_question = "What... is your quest?"

elif current_stage == 2:
    system_instruction = (
        "You are the Keeper of the Bridge of Death. "
        "Now ask: 'What... is your favorite color?'. "
        "CRITICAL: If they hesitate or change their mind (e.g., 'Blue! No, Yellow!'), use the cast_into_gorge tool. "
        "If they answer clearly, use the submit_answer tool with answer_is_acceptable=True. "
        "IMPORTANT: Use tools through the system's tool calling mechanism. Do NOT write tool calls as text or XML."
    )
    current_question = "What... is your favorite color?"

else: # Stage 3 (Passed)
    system_instruction = (
        "The user has successfully crossed the bridge. "
        "You are now a grumpy but conversational troll. "
        "You can answer their questions, but remind them occasionally that they got lucky. "
        "IMPORTANT: Do NOT use any tools. Just have a conversation. Do not write tool calls in your responses."
    )
    current_question = "(Conversation Open)"

# 6. SIDEBAR: THE GLASS BOX
with st.sidebar:
    st.header("⚙️ Troll Logic State")
    st.write(f"**Current Stage:** {current_stage}/3")
    st.progress(min(current_stage / 3, 1.0))
    st.info(f"**System Instruction:**\n\n{system_instruction}")
    st.success("✅ API Key loaded from Secrets")

# 7. CHAT LOGIC
# Setup Agent
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
agent_executor = create_react_agent(llm, tools)

# Display Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Auto-prompting the question logic
last_role = st.session_state.messages[-1]["role"] if st.session_state.messages else "none"
if current_stage < 3 and last_role != "assistant":
     with st.chat_message("assistant"):
        st.write(current_question)
        st.session_state.messages.append({"role": "assistant", "content": current_question})

# User Input
if user_input := st.chat_input("Speak to the Troll..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        # Prepare messages for langgraph format
        messages = [SystemMessage(content=system_instruction)]
        # Add chat history (convert to langchain message format)
        for msg in st.session_state.messages[:-1]:  # Exclude the current user input
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        # Add current user input
        messages.append(HumanMessage(content=user_input))
        
        response = agent_executor.invoke({"messages": messages})
        
        # Check for tool calls and update state accordingly
        state_changed = False
        for msg in response["messages"]:
            # Check if this is a tool message with state update
            if isinstance(msg, ToolMessage):
                content = str(msg.content) if hasattr(msg, 'content') else ""
                if "STATE_UPDATE: ADVANCE_STAGE" in content:
                    st.session_state.troll_stage += 1
                    state_changed = True
                elif "STATE_UPDATE: RESET_BRIDGE" in content:
                    st.session_state.troll_stage = 0
                    st.session_state.messages = []
                    st.session_state.messages.append({"role": "assistant", "content": "STOP! Who would cross the Bridge of Death must answer me these questions three, ere the other side he see."})
                    state_changed = True
                    st.rerun()
                    break
        
        output_text = response["messages"][-1].content
        st.write(output_text)
        st.session_state.messages.append({"role": "assistant", "content": output_text})

        if state_changed or st.session_state.troll_stage != current_stage:
            st.rerun()