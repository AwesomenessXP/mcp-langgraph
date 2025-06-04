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
    query: str
    step: int
    error: str
    current_answer: str

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
    return {"messages": [f"Error: {error_text}"], "error": error_text, "step": state.get("step", 1)}

# Node where the model decides to use the divide tool
async def reasoning_node(state: AgentState):
    print("reasoning_node\n\n")
    try:
        client = MultiServerMCPClient({
            "math": {
                "url": "http://127.0.0.1:8001/mcp",
                "transport": "streamable_http"
            }
        })
        tools = await client.get_tools()
        # Only keep the add tool
        add_tool = next(t for t in tools if t.name == "add")
        model_with_tools = model.bind_tools([add_tool])  # Only the add tool is available
        
        # Your custom prompt
        system_prompt = (
            "Step 1: add the first two numbers in the innermost parentheses, only use the add tool"
        )

        # Extract user message(s)
        user_msg = state["messages"][-1].content if isinstance(state["messages"], list) else state["messages"]

        # Compose message list (system prompt + user message)
        chat_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]

        response = await model_with_tools.ainvoke(chat_history)
        # print("query in reasoning_node", user_msg)
        # print("current_answer", response)
        return {
            "messages": [response], 
            "step": 1, 
            "query": user_msg, 
            "current_answer": "",
        }
    except Exception as e:
        return {
            "messages": [], 
            "step": 1, 
            "query": user_msg, 
            "current_answer": "", 
            "error": str(e)
        }
    
async def reasoning_node_2(state: AgentState):
    print("reasoning_node_2\n\n")
    try:
        client = MultiServerMCPClient({
            "math": {
                "url": "http://127.0.0.1:8001/mcp",
                "transport": "streamable_http"
            }
        })
        tools = await client.get_tools()
        # Only keep the add tool
        sub_tool = next(t for t in tools if t.name == "sub")
        model_with_tools = model.bind_tools([sub_tool])  # Only the add tool is available

        current_answer = state.get("current_answer", "")

        print("current_answer", current_answer)
        
        # Your custom prompt
        system_prompt = (
            "Subtract 8 from the current answer, use the subtract tool. Current answer: " + current_answer
        )

        print("system_prompt", system_prompt)

        # Extract user message(s)
        query = state.get("query", "")  

        # Compose message list (system prompt + user message)
        chat_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": current_answer}
        ]

        messages = state.get("messages", [])

        response = await model_with_tools.ainvoke(chat_history)
        return {
            "messages": messages + [response], 
            "step": 2, 
            "query": query, 
            "current_answer": ""
        }
    except Exception as e:
        return {
            "messages": messages, 
            "step": 2, 
            "query": query, 
            "current_answer": "", 
            "error": str(e)
        }
    
async def reasoning_node_3(state: AgentState):
    print("reasoning_node_3\n\n")
    try:
        client = MultiServerMCPClient({
            "math": {
                "url": "http://127.0.0.1:8001/mcp",
                "transport": "streamable_http"
            }
        })
        tools = await client.get_tools()
        # Only keep the add tool
        multiply_tool = next(t for t in tools if t.name == "multiply")
        model_with_tools = model.bind_tools([multiply_tool])  # Only the add tool is available

        current_answer = state.get("current_answer", "")
        print("current_answer", current_answer)
        # Your custom prompt
        system_prompt = (
            "Multiply the current answer by 6, use the multiply tool. Current answer: " + current_answer
        )

        print("system_prompt", system_prompt)

        # Extract user message(s)
        query = state.get("query", "")

        # Compose message list (system prompt + user message)
        chat_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": current_answer}
        ]

        messages = state.get("messages", [])

        response = await model_with_tools.ainvoke(chat_history)
        return {
            "messages": messages + [response], 
            "step": 3, 
            "query": query, 
            "current_answer": "",
        }
    except Exception as e:
        return {
            "messages": messages,
            "step": 3, 
            "query": query, 
            "current_answer": response.content, 
            "error": str(e)
        }

# Node to safely execute the divide tool
async def acting_node(state: AgentState):
    print("acting_node\n\n")
    try:
        client = MultiServerMCPClient({
            "math": {
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
            "messages": messages, 
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

# Build the LangGraph
async def build_graph():
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("reasoning_node", RunnableLambda(reasoning_node))
    graph_builder.add_node("reasoning_node_2", RunnableLambda(reasoning_node_2))
    graph_builder.add_node("reasoning_node_3", RunnableLambda(reasoning_node_3))
    graph_builder.add_node("safe_tools", RunnableLambda(acting_node))
    graph_builder.add_node("error_node", RunnableLambda(error_node))

    graph_builder.set_entry_point("reasoning_node")

    # Routing logic based on the presence of errors or tool calls
    def route(state: AgentState):
        # print("state", state)
        if "error" in state:
            return "error_node"
        messages = state.get("messages", [])
        if isinstance(messages, list) and messages:
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "safe_tools"
        return END

    graph_builder.add_conditional_edges("reasoning_node", route)
    graph_builder.add_conditional_edges("reasoning_node_2", route)
    graph_builder.add_conditional_edges("reasoning_node_3", route)
    
    # Route after safe_tools: end if success, error_node if error
    def route_from_safe_tools(state: AgentState):
        # print("state in route_from_safe_tools", state)
        error = state.get("error", None)
        if error:
            return "error_node"
        # Default to step 2 if not set
        step = state.get("step", 1)
        if step == 1:
            return "reasoning_node_2"
        elif step == 2:
            return "reasoning_node_3"
        else:
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
    response = asyncio.run(main(
        """
        Calculate the following expression:

        ((17 + 25) - 8) * 6

        what is the final answer?
        """))
    print('------------')
    print(response)

    # Write to a JSON file
    with open("response.json", "w") as f:
        json.dump(response, f, indent=2, default=serialize_message)