from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    collection_name="micro_collection",
    embedding_function=embeddings,
    persist_directory="chroma_db/micro",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

query = "what is economies of scale"
results = retriever.invoke(query)

print(f"Query: {query}\n")
print(f"Top {len(results)} results:\n")
for i, doc in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(f"Content:\n{doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print()
