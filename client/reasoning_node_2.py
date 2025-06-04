from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient 

from client.agent_state import AgentState

async def reasoning_node_2(state: AgentState):
    print("reasoning_node_2\n\n")
    try:
        client = MultiServerMCPClient({
            "insurance_compliance": {
                "url": "http://127.0.0.1:8001/mcp",
                "transport": "streamable_http"
            }
        })
        tools = await client.get_tools()
        # Initialize the model
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        model_with_tools = model.bind_tools(tools)  # Only the add tool is available

        current_answer = state.get("current_answer", "")

        print("current_answer", current_answer)
        
        # Your custom prompt
        system_prompt = (
            f"""
            Current answer: {current_answer}

            **Only use the analyze_summary tool.**

            Return a summary as JSON.
            """
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