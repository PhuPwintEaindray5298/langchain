import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

print("Initializing components ..............")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}
    Question: {question}
    Provide a detailed answer:
"""
)


def retrieval_chain_without_langchainExpressionLanguage(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.
    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step1: Retrieve relevant documents
    docs = retriever.invoke(query)
    # Step2: Format documents into context string
    context = format_docs(docs=docs)
    # Step3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)
    # Step4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)
    # Step5: Return the content
    return response.content


def retrieval_chain_with_langchain():
    """
    Create a retrieval chain using LCEL
    Returns a chain that can be invoked with {"question":"..."}

    Advantages:
    - Declarative and composable: easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of box
    - Built-in async: chain.invoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """

    # RunnablePassthrought -> the input for running this chain is unchange and going to be the original input
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print("Retrieving.....")

    query = "What is Pinecone in machine learning"

    # ==================================================
    # Option 0: Raw Invocation without RAG
    # ==================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # ==================================================
    # Option 1: Use Implementation without LCEL
    # ==================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: without Langchain Expression Language")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_langchainExpressionLanguage(
        query=query
    )
    print("\nAnswer")
    print(result_without_lcel)

    # ==================================================
    # Option 1: Use Implementation with LCEL
    # ==================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = retrieval_chain_with_langchain()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer")
    print(result_with_lcel)
