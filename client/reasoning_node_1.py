from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient 

from client.agent_state import AgentState

# Node where the model decides to use the divide tool
async def reasoning_node(state: AgentState):
    print("reasoning_node\n\n")
    try:
        client = MultiServerMCPClient({
            "insurance_compliance": {
                "url": "http://127.0.0.1:8001/mcp",
                "transport": "streamable_http"
            }
        })
        tools = await client.get_tools()

        # Initialize the model
        model = ChatOpenAI(model="gpt-4o", temperature=0)
        model_with_tools = model.bind_tools(tools)  # Only the add tool is available
        
        # Your custom prompt
        system_prompt = (
            """
            **Only use the extract_summary tool.**
            Return a summary as JSON.
            """
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
        print("current_answer", response)
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