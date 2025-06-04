from client.agent_state import AgentState

# Node to handle errors
def error_node(state: AgentState):
    print("error_node")
    messages = state.get("messages", [])
    error_text = None

    if isinstance(messages, list):
        # Find the last ToolMessage with status == "error"
        for msg in reversed(messages):
            status = getattr(msg, "status", None) or (msg.get("status") if isinstance(msg, dict) else None)
            if status == "error":
                # If using LangChain ToolMessage
                error_text = getattr(msg, "content", None)
                # If using dict ToolMessage
                if not error_text and isinstance(msg, dict):
                    error_text = msg.get("content")
                break

    # Fallback to state["error"] if we didn't find a ToolMessage error
    if not error_text:
        error_text = state.get("error", "Unknown error")

    print("⚠️ Error encountered:", error_text)
    return {"messages": [f"Error: {error_text}"], "error": error_text, "step": state.get("step", 1)}