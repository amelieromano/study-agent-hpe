import argparse
import glob
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

MODULE_MAP = {
    "micro":   "hpe_micro",
    "macro":   "hpe_macro",
    "history": "hpe_history",
    "up1":     "hpe_UP1",
    "up2":     "hpe_UP2",
}

def load_pdfs(folder, doc_type):
    docs = []
    pdf_files = glob.glob(os.path.join(folder, "*.pdf"))
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        for page in pages:
            page.metadata["doc_type"] = doc_type
        docs.extend(pages)
    return docs

def ingest_module(module_key):
    folder_name = MODULE_MAP[module_key]
    base = os.path.join("module_materials", folder_name)
    all_docs = []

    # Load course material subfolders
    for subfolder in ("lectures", "readings", "transcripts"):
        path = os.path.join(base, subfolder)
        if os.path.isdir(path):
            docs = load_pdfs(path, doc_type="course_material")
            print(f"  {subfolder}: {len(docs)} pages")
            all_docs.extend(docs)

    # Load tutorials
    tutorials_path = os.path.join(base, "tutorials")
    if os.path.isdir(tutorials_path):
        docs = load_pdfs(tutorials_path, doc_type="tutorial")
        print(f"  tutorials: {len(docs)} pages")
        all_docs.extend(docs)

    # Load history books (chapters and root PDFs per book)
    books_path = os.path.join(base, "books")
    if os.path.isdir(books_path):
        for book_dir in sorted(os.listdir(books_path)):
            book_path = os.path.join(books_path, book_dir)
            if not os.path.isdir(book_path):
                continue
            # Root-level PDFs (full book or extracts)
            docs = load_pdfs(book_path, doc_type="course_material")
            # Chapter-level PDFs
            chapters_path = os.path.join(book_path, "chapters")
            if os.path.isdir(chapters_path):
                docs += load_pdfs(chapters_path, doc_type="course_material")
            if docs:
                print(f"  books/{book_dir}: {len(docs)} pages")
            all_docs.extend(docs)

    # Load past papers
    past_papers_path = os.path.join("module_materials", "past_papers", module_key)
    if os.path.isdir(past_papers_path):
        docs = load_pdfs(past_papers_path, doc_type="past_paper")
        print(f"  past_papers: {len(docs)} pages")
        all_docs.extend(docs)
    else:
        print(f"  past_papers: folder not found, skipping")

    if not all_docs:
        print(f"No documents found for module '{module_key}'. Exiting.")
        return

    # Split — larger chunks for history to preserve argumentative prose flow
    if module_key == "history":
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    print(f"\nTotal chunks to index: {len(chunks)}")

    # Embed and store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_dir = os.path.join("chroma_db", module_key)
    collection_name = f"{module_key}_collection"

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )

    print(f"Done. {len(chunks)} chunks indexed into '{collection_name}' at '{persist_dir}'")


def ingest_documents(docs, module_key):
    """Chunk and embed a list of already-loaded Document objects into the module's ChromaDB collection.
    Returns the number of chunks added."""
    if not docs:
        return 0

    if module_key == "history":
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_dir = os.path.join("chroma_db", module_key)
    collection_name = f"{module_key}_collection"

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    vectorstore.add_documents(chunks)

    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest module PDFs into ChromaDB.")
    parser.add_argument(
        "--module",
        required=True,
        choices=["micro", "macro", "history", "up1", "up2", "all"],
        help="Module to ingest",
    )
    args = parser.parse_args()

    modules = list(MODULE_MAP.keys()) if args.module == "all" else [args.module]

    for module_key in modules:
        print(f"\n=== Ingesting module: {module_key} ===")
        ingest_module(module_key)


if __name__ == "__main__":
    main()
