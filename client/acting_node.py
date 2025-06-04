from langchain_mcp_adapters.client import MultiServerMCPClient 
from langgraph.prebuilt import ToolNode

from client.agent_state import AgentState

# Node to safely execute the divide tool
async def acting_node(state: AgentState):
    print("acting_node\n\n")
    try:
        client = MultiServerMCPClient({
            "insurance_compliance": {
                "url": "http://127.0.0.1:8001/mcp",
                "transport": "streamable_http"
            }
        })
        tools = await client.get_tools()
        tool_node = ToolNode(tools)

        messages = state.get("messages", [])

        # print("state in acting_node", state)
        result = await tool_node.ainvoke(state)
        print("result in acting_node", result)

        # Get previous messages and new tool messages
        step = state.get("step", 1)
        
        new_messages = result.get("messages", [])

        query = state.get("query", "")

        # return only the most relevant tool (the first tool)
        current_answer = new_messages[0].content if new_messages else ""

        # Check last message for error status
        if isinstance(new_messages, list) and new_messages:
            last_msg = new_messages[-1]
            # For LangChain ToolMessage, status may be an attribute or dict key
            status = getattr(last_msg, "status", None)
            if status == "error":
                return {
                    "messages": messages + [f"Error: {last_msg.content}"], 
                    "error": last_msg.content, 
                    "step": step, 
                    "query": query, 
                    "current_answer": current_answer
                }
        # print("acting_node result", result)
        return {
            "messages": messages + new_messages, 
            "step": step, 
            "query": query, 
            "current_answer": current_answer
        }
    except Exception as e:
        return {
            "messages": messages,
            "step": step,
            "query": query,
            "current_answer": current_answer,
            "error": str(e)
        }