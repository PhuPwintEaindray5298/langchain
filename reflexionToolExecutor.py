from dotenv import load_dotenv

load_dotenv()

from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from reflexionSchema import AnswerQuestion, ReviseAnswer

tavily_tool = TavilySearch(max_results=3)


def run_queries(search_question: list[str], **kwargs):
    """Run the generated queries"""
    return tavily_tool.batch({"query": query} for query in search_question)


execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)
