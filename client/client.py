import asyncio
import json
from typing import TypedDict, Union, List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# Define the state schema
class AgentState(TypedDict, total=False):
    messages: Union[str, BaseMessage, List[BaseMessage]]
    error: str

# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Serializer for JSON output
def serialize_message(obj):
    if isinstance(obj, BaseMessage):
        return obj.dict()
    elif isinstance(obj, Exception):
        return str(obj)
    return obj

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
    return {"messages": [f"Error: {error_text}"], "error": error_text}

# Node where the model decides to use the divide tool
async def divide_node(state: AgentState):
    print("divide_node")
    try:
        client = MultiServerMCPClient({
            "math": {
                "url": "http://127.0.0.1:8001/mcp",
                "transport": "streamable_http"
            }
        })
        tools = await client.get_tools()
        model_with_tools = model.bind_tools(tools)
        response = await model_with_tools.ainvoke(state["messages"])
        print("divide_node response", response)
        return {"messages": [response]}
    except Exception as e:
        return {"error": str(e)}

# Node to safely execute the divide tool
async def safe_tool_runner(state: AgentState):
    print("safe_tool_runner")
    try:
        client = MultiServerMCPClient({
            "math": {
                "url": "http://127.0.0.1:8001/mcp",
                "transport": "streamable_http"
            }
        })
        tools = await client.get_tools()
        tool_node = ToolNode(tools)
        result = await tool_node.ainvoke(state)
        print("safe_tool_runner result", result)
        return result
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}

# Build the LangGraph
async def build_graph():
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("divide_node", RunnableLambda(divide_node))
    graph_builder.add_node("safe_tools", RunnableLambda(safe_tool_runner))
    graph_builder.add_node("error_node", RunnableLambda(error_node))

    graph_builder.set_entry_point("divide_node")

    # Routing logic based on the presence of errors or tool calls
    def route(state: AgentState):
        print("state", state)
        if "error" in state:
            return "error_node"
        messages = state.get("messages", [])
        if isinstance(messages, list) and messages:
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "safe_tools"
        return END

    graph_builder.add_conditional_edges("divide_node", route)
    
    # Route after safe_tools: end if success, error_node if error
    def route_from_safe_tools(state: AgentState):
        print("route_from_safe_tools", state)
        messages = state.get("messages", [])
        # Check last message for error status
        if isinstance(messages, list) and messages:
            last_msg = messages[-1]
            # For LangChain ToolMessage, status may be an attribute or dict key
            status = getattr(last_msg, "status", None)
            # Or, if using dicts:
            # status = last_msg.get("status")
            if status == "error":
                return "error_node"
        return END
        
    graph_builder.add_conditional_edges("safe_tools", route_from_safe_tools)

    # After error_node, always end
    graph_builder.add_edge("error_node", END)

    return graph_builder.compile()

# Main function to run the graph
async def main(query: str):
    graph = await build_graph()
    response = await graph.ainvoke({"messages": query})
    return response

if __name__ == "__main__":
    response = asyncio.run(main("Multiply 64 by 4 using the multiply tool"))
    print('------------')
    print(response)

    # Write to a JSON file
    with open("response.json", "w") as f:
        json.dump(response, f, indent=2, default=serialize_message)