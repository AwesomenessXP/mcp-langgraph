import asyncio
import json
import argparse

from dotenv import load_dotenv
from client.state_machine import build_graph, serialize_message

load_dotenv()

# Main function to run the graph
async def main(query: str):
    graph = await build_graph()
    response = await graph.ainvoke({"messages": query})
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a COI JSON file.")
    parser.add_argument("json_path", help="Path to the COI JSON file")
    args = parser.parse_args()

    # Load the COI document from the specified file path
    with open(args.json_path, "r") as f:
        coi = json.load(f)

    response = asyncio.run(main(f"{coi}"))
    print('------------')
    print(serialize_message(response.get("current_answer", "")))

    # Write to a JSON file
    with open("response.json", "w") as f:
        json.dump(response, f, indent=2, default=serialize_message)