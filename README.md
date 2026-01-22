# 🧌 The Bridge of Death - AI Agent Demo

> **None shall pass...** without answering these questions three!

An interactive AI agent demo that brings the legendary Bridge of Death from Monty Python to life. Cross the bridge by answering the troll's three questions, or face the Gorge of Eternal Peril! This project demonstrates how to build a deterministic, state-managed conversational AI agent using LangChain, LangGraph, and Groq.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.0.0+-FF6B6B?style=flat)](https://www.langchain.com/)

## 🎭 What Is This?

This demo wraps a **rigid deterministic workflow** in a playful persona. The Troll's mood swings are actually gated by **verified tool executions**, ensuring that no user can "hallucinate" their way across the bridge without permission. It's a perfect example of how to build AI agents that maintain strict business logic while providing an engaging user experience.

## ✨ Features

- 🧌 **Interactive Bridge Troll** - Face-to-face conversation with an AI-powered Monty Python troll
- 🛡️ **State Management** - Deterministic workflow with verified tool executions
- 🔒 **Security Guardrails** - Protection against prompt injection attacks
- ⚡ **Fast Responses** - Powered by Groq's lightning-fast LLM inference
- 🎯 **Tool-Based Governance** - Questions are gated by actual tool calls, not just LLM responses
- 🔥 **Gorge of Eternal Peril** - Fail to answer correctly and face the consequences!

## 🛠️ Tech Stack

- **Streamlit** - Web interface and UI framework
- **LangChain** - LLM orchestration and tool management
- **LangGraph** - Agent execution and state management
- **Groq** - High-performance LLM inference
- **Python 3.8+** - Core language

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- A Groq API key ([Get one free here](https://console.groq.com/))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mattcoelho/LaAgent.git
   cd LaAgent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**
   
   Create a `.streamlit/secrets.toml` file:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   
   Navigate to `http://localhost:8501` and prepare to face the troll!

## ☁️ Deployment

### Streamlit Cloud

1. **Push your code to GitHub**
   ```bash
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Connect your GitHub repository
   - Add your `GROQ_API_KEY` in the app's Settings → Secrets
   - Deploy!

3. **Your app is live!**
   
   Share the URL with friends and watch them attempt to cross the bridge!

## 🏗️ How It Works

### Architecture

The application uses a **state-driven agent architecture**:

1. **State Management** - The troll's stage (0: Name, 1: Quest, 2: Color, 3: Passed, -1: Failed) is tracked in Streamlit's session state
2. **Dynamic System Prompts** - Each stage has a specific system prompt that guides the LLM's behavior
3. **Tool-Based Verification** - The `submit_answer()` tool must be called with `answer_is_acceptable=True` to advance stages
4. **Failure Handling** - The `cast_into_gorge()` tool triggers the failure state when users answer incorrectly

### Key Components

- **Tools** - `submit_answer()` and `cast_into_gorge()` control state transitions
- **System Prompts** - Stage-specific instructions that prevent hallucinations and enforce security
- **Message History** - Conversation context is maintained throughout the interaction
- **Error Handling** - Rate limiting and API errors are gracefully handled

### Security Features

- **Prompt Injection Protection** - System prompts explicitly ignore attempts to change instructions
- **Tool Verification** - State changes only occur through verified tool executions
- **Response Filtering** - Completion messages are filtered to prevent premature success

## 🎮 Usage

1. **Start the conversation** - The troll will immediately demand your name
2. **Answer the three questions**:
   - What is your name?
   - What is your quest?
   - What is your favorite color?
3. **Cross the bridge** - Answer all three correctly to pass
4. **Or face the Gorge** - Hesitate, change your mind, or answer incorrectly and be cast into the Gorge of Eternal Peril!

## 📝 Project Structure

```
LaAgent/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Share your own bridge troll variations!

## 📄 License

This project is open source and available for educational and demonstration purposes.

## 🙏 Acknowledgments

- Inspired by Monty Python and the Holy Grail
- Built with [LangChain](https://www.langchain.com/) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [Groq](https://groq.com/) for fast inference
- Deployed with [Streamlit](https://streamlit.io/)

---

**Remember**: None shall pass without answering these questions three! 🧌
