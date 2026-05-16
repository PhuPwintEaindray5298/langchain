from typing import Any, Dict

from agentic_graph.chains.generation import generation_chain
from agentic_graph.state import GraphState

def generate(state: GraphState) -> Dict[str, Any]:
    print("--Generate--")
    question = state['question']
    documents = state['documents']
    generation = generation_chain.invoke({"context":documents,"questions":question})
    return {"documents":documents,"question":question,"generation":generation}