from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

from client.reasoning_node_1 import reasoning_node
from client.reasoning_node_2 import reasoning_node_2
from client.reasoning_node_3 import reasoning_node_3
from client.acting_node import acting_node
from client.error_node import error_node

from client.agent_state import AgentState

# Serializer for JSON output
def serialize_message(obj):
    if isinstance(obj, BaseMessage):
        return obj.dict()
    elif isinstance(obj, Exception):
        return str(obj)
    return obj

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