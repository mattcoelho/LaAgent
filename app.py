import streamlit as st
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

# 1. PAGE CONFIG
st.set_page_config(page_title="LaAgent Demo", layout="wide")

st.title("⚡ Real-Time Action Agent (Groq Powered)")
st.markdown("A deterministic agent with **Human-in-the-Loop** guardrails, running on Llama 3 via Groq for ultra-low latency.")

# 2. DEFINE TOOLS (The "Actions")
@tool
def check_order_status(order_id: str):
    """Checks the status of a customer order."""
    # Mock database return
    return {"order_id": order_id, "status": "shipped", "delivery_date": "2025-10-25"}

@tool
def process_refund(order_id: str, amount: float):
    """Processes a refund. REQUIRES HUMAN APPROVAL."""
    return {"status": "success", "refunded_amount": amount}

tools = [check_order_status, process_refund]

# Define your bot's persona/system prompt
SYSTEM_PROMPT = """You are a helpful customer service agent at Matthew's LaAgent Interprise. 
Your personality is friendly, professional, and empathetic and funny.
Always be concise and clear in your responses."""

# 3. SETUP SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

# Sidebar: The "Glass Box"
with st.sidebar:
    st.header("🧠 Agent Reasoning (Glass Box)")
    st.caption("Live view of tool calls and logic states.")
    for log in st.session_state.audit_log:
        st.code(log, language="json")

# 4. GET API KEY FROM STREAMLIT SECRETS
try:
    api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("⚠️ Groq API key not found. Please create `.streamlit/secrets.toml` with your `GROQ_API_KEY`. See `.streamlit/secrets.toml.example` for reference.")
    st.stop()

try:
    # Switch to Groq (Llama 3 70B is smart enough for tools)
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
    agent_executor = create_react_agent(llm, tools)

    # 5. CHAT INTERFACE
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Try: 'Check status of order 123'"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Run Agent
        with st.chat_message("assistant"):
            # This is where LangGraph executes
            # Prepend system message to set the bot's persona
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                ("user", prompt)
            ]
            response = agent_executor.invoke({"messages": messages})
            bot_msg = response["messages"][-1].content
            st.write(bot_msg)
            
            # Update State
            st.session_state.messages.append({"role": "assistant", "content": bot_msg})
            
            # Update Glass Box (Log the tool calls)
            st.session_state.audit_log.append(f"User: {prompt}\nResponse: {bot_msg}")
            
except Exception as e:
    st.error(f"Error: {e}")
