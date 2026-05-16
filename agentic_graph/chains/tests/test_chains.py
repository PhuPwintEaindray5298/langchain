from dotenv import load_dotenv

load_dotenv()

from agentic_graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from agentic_rag_ingestion import retriever
from agentic_graph.chains.generation import generation_chain

def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_test = docs[1].page_content
    res: GradeDocuments = retrieval_grader.invoke(
        {"question":question,"document":doc_test}
    )
    assert res.binary_score == "yes"

def test_retrieval_grader_answer_no() -> None:
    question = "how to make pizza"
    docs = retriever.invoke(question)
    doc_test = docs[1].page_content
    res: GradeDocuments = retrieval_grader.invoke(
        {"question":question,"document":doc_test}
    )
    assert res.binary_score == "no"

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context":docs,"question":question})
    print(generation)