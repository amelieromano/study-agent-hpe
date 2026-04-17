import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

LECTURES_DIR = "module_materials/hpe_micro/lectures"

pdf_files = glob.glob(f"{LECTURES_DIR}/*.pdf")
if not pdf_files:
    print(f"No PDF files found in {LECTURES_DIR}")
    exit(1)

pdf_path = pdf_files[0]
print(f"Loading: {pdf_path}\n")

loader = PyPDFLoader(pdf_path)
pages = loader.load()

print(f"Total pages loaded: {len(pages)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

print(f"Total chunks produced: {len(chunks)}")

print("\n--- First 3 chunks ---")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n[Chunk {i + 1}]")
    print(f"Content:\n{chunk.page_content}")
    print(f"Metadata: {chunk.metadata}")
