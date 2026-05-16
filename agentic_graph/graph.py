from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from agentic_graph.consts import RETREIVE,GENERATE,GRADE_DOCUMENTS,WEBSEARCH
from agentic_graph.nodes import generate, grade_documents, retrieve, web_search
from agentic_graph.state import GraphState
from agentic_graph.chains.answer_grader import answer_grader
from agentic_graph.chains.hallucination_grader import hallucination_grader

load_dotenv()

def decide_to_generate(state):
    print("---Assess Graded Documents ---")
    if state["web_search"]:
        print(
            "---Decision: Not all documents are not relevant to questions"
        )
        return WEBSEARCH
    else:
        print("--Decision: Generate --")
        return GENERATE

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("--Check Hallucinations --")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents":documents, "generation":generation}
    )

    if hallucination_grader := score.binary_score:
        print("---Decision: Generation is grouonded in documents ---")
        print("---Grade Generation vs Question ---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("--Decision: Generation is not grouonded in documents, Re-try")
        return "not supported"
    

workflow = StateGraph(GraphState)
workflow.add_node(RETREIVE,retrieve)
workflow.add_node(GRADE_DOCUMENTS,grade_documents)
workflow.add_node(WEBSEARCH,web_search)
workflow.add_node(GENERATE,generate)
workflow.set_entry_point(RETREIVE)
workflow.set_edge(RETREIVE,GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS,decide_to_generate,{WEBSEARCH:WEBSEARCH,GENERATE:GENERATE})
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)
workflow.add_edge(WEBSEARCH,GENERATE)
workflow.add_edge(GENERATE,END)

graph = workflow.compile()
print(graph.get_graph().draw_mermaid())