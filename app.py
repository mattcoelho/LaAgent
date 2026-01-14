import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# 1. PAGE CONFIG (Looks Professional)
st.set_page_config(page_title="Sierra Demo Agent", layout="wide")

st.title("🛡️ Enterprise Action Agent Demo")
st.markdown("A deterministic agent with **Human-in-the-Loop** guardrails.")

# 2. DEFINE TOOLS ( The "Actions")
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

# 3. SETUP SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

# Sidebar: The "Glass Box" (Sierra loves transparency)
with st.sidebar:
    st.header("🧠 Agent Reasoning (Glass Box)")
    st.caption("Live view of tool calls and logic states.")
    for log in st.session_state.audit_log:
        st.code(log, language="json")

# 4. API KEY INPUT (Secure way for public demo)
api_key = st.text_input("Enter OpenAI API Key to test:", type="password")

if api_key:
    llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
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
            try:
                # This is where LangGraph executes
                response = agent_executor.invoke({"messages": [("user", prompt)]})
                bot_msg = response["messages"][-1].content
                st.write(bot_msg)
                
                # Update State
                st.session_state.messages.append({"role": "assistant", "content": bot_msg})
                
                # Update Glass Box (Log the tool calls)
                # In a real app, you'd parse the intermediate steps here
                st.session_state.audit_log.append(f"User: {prompt}\nResponse: {bot_msg}")
                
            except Exception as e:
                st.error(f"Error: {e}")