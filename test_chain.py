from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    collection_name="micro_collection",
    embedding_function=embeddings,
    persist_directory="chroma_db/micro",
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a UCL university study assistant helping students understand their course material.

Use ONLY the context below to answer the question. Do not use any outside knowledge.
If the answer is not found in the provided material, say clearly: "This topic is not covered in the uploaded course materials."
Where relevant, cite the source document name from the metadata.

Context:
{context}

Question: {question}

Answer:""",
)


def print_result(label, result):
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"Answer:\n{result['result']}\n")
    source_files = list({doc.metadata.get("source", "unknown") for doc in result["source_documents"]})
    print("Sources used:")
    for source in source_files:
        print(f"  - {source}")


query = "Explain economies of scale"
print(f"Question: {query}")

# Default prompt
default_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)
default_result = default_chain.invoke({"query": query})
print_result("DEFAULT PROMPT", default_result)

# Custom prompt
custom_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": CUSTOM_PROMPT},
)
custom_result = custom_chain.invoke({"query": query})
print_result("CUSTOM PROMPT (UCL Study Assistant)", custom_result)
