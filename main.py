import asyncio
import json

from dotenv import load_dotenv
from client.state_machine import build_graph, serialize_message

load_dotenv()

# Main function to run the graph
async def main(query: str):
    graph = await build_graph()
    response = await graph.ainvoke({"messages": query})
    return response

if __name__ == "__main__":
    response = asyncio.run(main(
        """
        {
        "policy_number": "123456789",
        "holder": { "name": "Alice Smith", "age": 37 },
        "coverage": [
            { "type": "life", "amount": 150000, "premium": 850 },
            { "type": "health", "amount": 300000, "premium": 1200 }
        ],
        "start_date": "2022-01-01",
        "end_date": "2027-01-01",
        "claims": [
            { "date": "2023-04-10", "type": "health", "amount": 2000 }
        ]
        }
        """))
    print('------------')
    print(serialize_message(response))

    # Write to a JSON file
    with open("response.json", "w") as f:
        json.dump(response, f, indent=2, default=serialize_message)