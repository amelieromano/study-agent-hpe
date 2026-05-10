"""
build_equation_sheet.py

Loads every chunk from the micro ChromaDB collection, filters for chunks that
contain mathematical content, batches them through the LLM for extraction, and
compiles a structured markdown table of every equation into equation_sheet_micro.md.

Deduplication is done in Python (not via LLM) to avoid model repetition loops.
"""

import re
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Load all chunks from micro ChromaDB collection
# ---------------------------------------------------------------------------
print("Loading micro ChromaDB collection...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="micro_collection",
    embedding_function=embeddings,
    persist_directory="chroma_db/micro",
)
result = vectorstore._collection.get()
all_texts = result["documents"]
print(f"  Total chunks in collection: {len(all_texts)}")

# ---------------------------------------------------------------------------
# 2. Filter for equation-relevant chunks
# ---------------------------------------------------------------------------
MATH_RE = re.compile(
    r'equation|formula|equals?|calculat'           # keywords
    r'|where\s+[A-Za-z]'                           # "where X is..."
    r'|\b(TR|TC|MC|MR|MP|AP|MU|MRS|MRT|TRS'       # econ shorthands before =
    r'|GDP|CPI|CS|PS|DWL|VC|FC|AC|AVC|AFC'
    r'|PED|PES|YED|XED)\s*='
    r'|[∑∫∂π≤≥≠αβγδεζθλμσφψωΔΠΣ]'               # unicode math
    r'|\\frac|\\sum|\\int|\\partial'               # LaTeX
    r'|\^[{(]|_[{(]'                               # superscript/subscript
    r'|[A-Z]\s*=\s*[A-Z0-9(]'                      # "X = Y..." patterns
    r'|=\s*[A-Z0-9(\\]',                           # assignment-like =
    re.IGNORECASE,
)

filtered = [t for t in all_texts if MATH_RE.search(t)]
print(f"  Equation-relevant chunks after filtering: {len(filtered)}")

# ---------------------------------------------------------------------------
# 3. Batch filtered chunks through LLM for equation extraction
# ---------------------------------------------------------------------------
BATCH_SIZE = 15

EXTRACTION_PROMPT = """\
You are an expert microeconomics tutor extracting equations from raw study material.

Scan the text below and identify every distinct mathematical equation or formula.
For each one, output a single markdown table row (no header, no preamble, no explanation):

| Equation Name | Formula | Variables Defined | Topic Area | Related Equations |

Rules:
- Equation Name: a clear descriptive name (e.g. "Total Revenue", "Price Elasticity of Demand")
- Formula: the equation using clean notation (e.g. TR = P × Q, MR = dTR/dQ)
- Variables Defined: short definition of each symbol used (e.g. "P = price, Q = quantity")
- Topic Area: one of: Consumer Theory | Production | Costs | Market Structure | Welfare | Factor Markets | General Equilibrium | Other
- Related Equations: comma-separated names of directly related equations (can be empty)

Only output table rows. If a chunk has no real equations, output nothing.
Do not repeat the same equation twice within this batch.

TEXT:
{context}
"""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, request_timeout=60)

raw_rows = []  # list of parsed row strings
num_batches = (len(filtered) + BATCH_SIZE - 1) // BATCH_SIZE

MAX_RETRIES = 3

for i in range(0, len(filtered), BATCH_SIZE):
    batch_num = i // BATCH_SIZE + 1
    batch = filtered[i : i + BATCH_SIZE]
    print(f"  Batch {batch_num}/{num_batches} ({len(batch)} chunks)...", end=" ", flush=True)

    context = "\n\n---\n\n".join(batch)
    text = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = llm.invoke([{"role": "user", "content": EXTRACTION_PROMPT.format(context=context)}])
            text = response.content.strip()
            break
        except Exception as e:
            print(f"\n    [attempt {attempt}/{MAX_RETRIES} failed: {e}]", end=" ", flush=True)
            if attempt == MAX_RETRIES:
                print("skipping batch.")

    # Parse only lines that look like table rows: start and end with |
    new_rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|") and line.endswith("|") and "---|" not in line:
            # Skip any accidentally included header lines
            cols = [c.strip() for c in line.strip("|").split("|")]
            if len(cols) == 5 and cols[0].lower() not in ("equation name", "name"):
                new_rows.append(line)

    raw_rows.extend(new_rows)
    print(f"extracted {len(new_rows)} row(s)")

print(f"\nTotal raw rows before deduplication: {len(raw_rows)}")

# ---------------------------------------------------------------------------
# 4. Python-level deduplication + sort
#    Key: normalised formula string (strips spaces, lowercases)
# ---------------------------------------------------------------------------
def parse_row(row: str):
    cols = [c.strip() for c in row.strip("|").split("|")]
    if len(cols) == 5:
        return cols  # [name, formula, variables, topic, related]
    return None

seen_formulas = set()
unique_rows = []

for row in raw_rows:
    parsed = parse_row(row)
    if parsed is None:
        continue
    formula_key = re.sub(r'\s+', '', parsed[1]).lower()
    if formula_key and formula_key not in seen_formulas:
        seen_formulas.add(formula_key)
        unique_rows.append(parsed)

# Sort by Topic Area, then Equation Name
TOPIC_ORDER = [
    "Consumer Theory", "Production", "Costs", "Market Structure",
    "Welfare", "Factor Markets", "General Equilibrium", "Other",
]
def sort_key(row):
    topic = row[3]
    order = TOPIC_ORDER.index(topic) if topic in TOPIC_ORDER else 99
    return (order, row[0].lower())

unique_rows.sort(key=sort_key)
print(f"Final unique equations after deduplication: {len(unique_rows)}")

# ---------------------------------------------------------------------------
# 5. Write output file
# ---------------------------------------------------------------------------
output_path = "equation_sheet_micro.md"

with open(output_path, "w", encoding="utf-8") as f:
    f.write("# Microeconomics Equation Sheet\n\n")
    f.write(
        f"*Generated from {len(filtered)} equation-relevant chunks "
        f"out of {len(all_texts)} total chunks. "
        f"{len(unique_rows)} unique equations compiled.*\n\n"
    )
    f.write("| Equation Name | Formula | Variables Defined | Topic Area | Related Equations |\n")
    f.write("|---|---|---|---|---|\n")
    for row in unique_rows:
        f.write("| " + " | ".join(row) + " |\n")

print(f"Saved to {output_path}")
