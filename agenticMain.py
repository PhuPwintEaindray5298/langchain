from dotenv import load_dotenv
load_dotenv()

from agentic_graph.graph import graph

if __name__ == '__main__':
    print("Hello Agentic RAG")
    print(graph.invoke(input={"question":"what is agent memory?"}))