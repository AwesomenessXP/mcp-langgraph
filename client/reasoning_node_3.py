from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient 

from client.agent_state import AgentState

async def reasoning_node_3(state: AgentState):
    print("reasoning_node_3\n\n")
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

        current_answer = state.get("current_answer", "")
        print("current_answer", current_answer)
        # Your custom prompt
        system_prompt = (
            f"""
            Format the current answer as an email to joesimile@gmail.com. 
            
            **Only use the format_email tool.**

            Current answer: {current_answer}

            Format the output as a professional email with a subject line, greeting, body, and closing, all as a single string.
            Do not use markdown or code blocks—output only the email as it would be sent.
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