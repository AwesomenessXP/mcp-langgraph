from typing import TypedDict, Union, List
from langchain_core.messages import BaseMessage

# Define the state schema
class AgentState(TypedDict, total=False):
    messages: Union[str, BaseMessage, List[BaseMessage]]
    query: str
    step: int
    error: str
    current_answer: str