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
        "document_id": "coi456",
        "pages": [
            {
            "page_number": 1,
            "tables": [
                {
                "title": "Coverage Table",
                "rows": [
                    {
                    "Type of Insurance": "General Liability",
                    "Policy Number": "GL-222222",
                    "Effective Date": "2024-05-01",
                    "Expiration Date": "2025-05-01",
                    "Limits": "$2,000,000 per occurrence"
                    },
                    {
                    "Type of Insurance": "Automobile Liability",
                    "Policy Number": "",
                    "Effective Date": "",
                    "Expiration Date": "",
                    "Limits": ""
                    }
                ]
                }
            ]
            },
            {
            "page_number": 2,
            "tables": [
                {
                "type": "key_value",
                "key_value_data": [
                    { "key": "Workers' Compensation Policy", "value": "WC-888888" },
                    { "key": "Workers' Compensation Effective Date", "value": "2024-05-01" },
                    { "key": "Workers' Compensation Expiry Date", "value": "2025-05-01" },
                    { "key": "Umbrella Liability Aggregate", "value": "$5,000,000" }
                ]
                }
            ]
            }
        ]
        }
        """))
    print('------------')
    print(serialize_message(response.get("current_answer", "")))

    # Write to a JSON file
    with open("response.json", "w") as f:
        json.dump(response, f, indent=2, default=serialize_message)