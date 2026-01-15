import streamlit as st
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
import json
import os
from datetime import datetime

# 1. PAGE CONFIG
st.set_page_config(page_title="LaAgent Demo", layout="wide")

st.title("⚡ Real-Time Action Agent (Groq Powered)")
st.markdown("A deterministic agent with **Human-in-the-Loop** guardrails, running on Llama 3 via Groq for ultra-low latency.")

# 2. DEFINE TOOLS (The "Actions")
@tool
def check_order_status(order_id: str):
    """Checks the status of a customer order."""
    # #region agent log
    debug_log("app.py:15", "check_order_status called", {"order_id": order_id}, "B")
    # #endregion
    # Mock database return
    result = {"order_id": order_id, "status": "shipped", "delivery_date": "2025-10-25"}
    # #region agent log
    debug_log("app.py:18", "check_order_status returning", {"result": result}, "B")
    # #endregion
    return result

@tool
def process_refund(order_id: str, amount: float):
    """Processes a refund. REQUIRES HUMAN APPROVAL."""
    return {"status": "success", "refunded_amount": amount}

tools = [check_order_status, process_refund]

# Debug logging helper
def debug_log(location, message, data, hypothesis_id=None):
    log_path = "/Users/teusbombs/Documents/Development/LaAgent/.cursor/debug.log"
    try:
        with open(log_path, "a") as f:
            log_entry = {
                "id": f"log_{int(datetime.now().timestamp() * 1000)}",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "location": location,
                "message": message,
                "data": data,
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": hypothesis_id
            }
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass

# #region agent log
debug_log("app.py:25", "Tools defined", {"tool_count": len(tools), "tool_names": [t.name for t in tools]}, "C")
# #endregion

# Define your bot's persona/system prompt
SYSTEM_PROMPT = """You are a troll that is blocking a bridge.
Before you do anything helpful the user must answer these three questions.
1. What is your name?
2. What is your quest?
3. What is your favorite color?

If the user answers correctly, you will help them cross the bridge.
If the user answers incorrectly, you will not help them cross the bridge.
"""

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

# 4. GET API KEY FROM STREAMLIT SECRETS (with fallback for local development)
try:
    api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, AttributeError):
    # Fallback for local development or if secret not configured in Streamlit Cloud
    st.info("💡 **Note:** For Streamlit Cloud deployment, configure `GROQ_API_KEY` in your app's Settings → Secrets. For local testing, enter your key below.")
    api_key = st.text_input("Enter Groq API Key:", type="password", help="Get a free key at console.groq.com")

if api_key:
    try:
        # Switch to Groq (Llama 3 70B is smart enough for tools)
        # #region agent log
        debug_log("app.py:62", "Creating LLM", {"model": "llama-3.3-70b-versatile", "has_api_key": bool(api_key)}, "D")
        # #endregion
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
        # #region agent log
        debug_log("app.py:64", "Creating agent executor", {"tool_count": len(tools)}, "E")
        # #endregion
        agent_executor = create_react_agent(llm, tools)

        # 5. CHAT INTERFACE
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input("Ask me anything"):
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
                # #region agent log
                debug_log("app.py:81", "Messages before invoke", {"message_count": len(messages), "system_prompt": SYSTEM_PROMPT[:100], "user_prompt": prompt}, "A")
                # #endregion
                # #region agent log
                debug_log("app.py:84", "Invoking agent executor", {}, "E")
                # #endregion
                response = agent_executor.invoke({"messages": messages})
                # #region agent log
                debug_log("app.py:85", "Agent response received", {"response_type": type(response).__name__, "message_count": len(response.get("messages", [])) if isinstance(response, dict) else 0}, "E")
                # #endregion
                # #region agent log
                if isinstance(response, dict) and "messages" in response:
                    last_msg = response["messages"][-1] if response["messages"] else None
                    debug_log("app.py:86", "Last message details", {"message_type": type(last_msg).__name__ if last_msg else None, "has_content": hasattr(last_msg, "content") if last_msg else False, "content_preview": str(last_msg.content)[:200] if last_msg and hasattr(last_msg, "content") else None}, "A")
                # #endregion
                bot_msg = response["messages"][-1].content
                # #region agent log
                debug_log("app.py:87", "Bot message extracted", {"message_length": len(str(bot_msg)) if bot_msg else 0, "message_preview": str(bot_msg)[:200] if bot_msg else None}, "A")
                # #endregion
                st.write(bot_msg)
                
                # Update State
                st.session_state.messages.append({"role": "assistant", "content": bot_msg})
                
                # Update Glass Box (Log the tool calls)
                st.session_state.audit_log.append(f"User: {prompt}\nResponse: {bot_msg}")
                
    except Exception as e:
        # #region agent log
        debug_log("app.py:95", "Exception caught", {"error_type": type(e).__name__, "error_message": str(e), "error_args": str(e.args) if hasattr(e, "args") else None}, "A")
        # #endregion
        st.error(f"Error: {e}")
