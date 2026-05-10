# Year 1 HPE Study Agent

A domain-specific RAG application built to support my UCL Year 1 History, Politics and Economics degree. I built it because generic AI assistants hallucinate freely when asked about specific lecture content, and because I wanted to understand — by building — where retrieval-augmented generation actually adds value over a prompt-stuffed LLM. The result is a multi-modal study tool that keeps answers grounded in my uploaded course materials, separates retrieval strategies by academic discipline, and surfaces exam-relevant patterns from past papers when they exist in the index.

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │             INGESTION PIPELINE           │
                        │                                          │
  PDFs (lectures,       │  PyPDFLoader → RecursiveCharacterText    │
  readings, past        │  Splitter (500/50 econ, 800/100 hist) →  │
  papers, uploads)  ──► │  tag doc_type metadata →                 │
                        │  OpenAIEmbeddings (text-embedding-3-small)│
                        │  → ChromaDB (one collection per module)  │
                        └──────────────────┬──────────────────────┘
                                           │ persisted to chroma_db/[module]
                                           ▼
                        ┌─────────────────────────────────────────┐
                        │            VECTOR STORE LAYER            │
                        │                                          │
                        │   micro_collection   (chroma_db/micro)  │
                        │   macro_collection   (chroma_db/macro)  │
                        │   history_collection (chroma_db/history)│
                        │   up1_collection     (chroma_db/up1)    │
                        │   up2_collection     (chroma_db/up2)    │
                        └──────────────────┬──────────────────────┘
                                           │ similarity search (k=5–15)
                                           ▼
                        ┌─────────────────────────────────────────┐
                        │              QUERY PIPELINE              │
                        │                                          │
  User question     ──► │  StudyAgent.query(question, mode)       │
                        │       │                                  │
                        │       ├─ mode guard (history/quant only) │
                        │       ├─ select k by mode complexity     │
                        │       ├─ mode-specific PromptTemplate    │
                        │       └─ RetrievalQA chain               │
                        │             │                            │
                        │    OpenAI GPT-4o-mini (temp=0.3)        │
                        │             │                            │
                        │    return answer + source metadata       │
                        └──────────────────┬──────────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────────┐
                        │           STREAMLIT FRONTEND             │
                        │                                          │
                        │  • Module selector + mode selector       │
                        │  • Answer card with inline source tags   │
                        │  • Confidence rating → Weak Spots tab    │
                        │  • Exam countdown + study time estimator │
                        │  • Upload Notes → live re-ingestion      │
                        │  • Markdown export for key modes         │
                        └─────────────────────────────────────────┘
```

---

## Stack

| Layer | Technology | Why |
|---|---|---|
| Embeddings | `text-embedding-3-small` (OpenAI) | Best cost/quality trade-off for retrieval; 1536 dimensions, $0.02/1M tokens |
| LLM | `gpt-4o-mini` (temp 0.3) | Accurate, cheap, fast; temperature kept low because study answers should be stable |
| Vector store | ChromaDB (local, persisted) | Zero-infrastructure for a single-user app; trivially swappable for Pinecone/Weaviate |
| Orchestration | LangChain (`RetrievalQA`, `PromptTemplate`) | Gives clean separation between retrieval and generation without hiding the pipeline |
| Ingestion | `PyPDFLoader` + `RecursiveCharacterTextSplitter` | Recursive splitter degrades gracefully across paragraph → sentence → word boundaries |
| Frontend | Streamlit | Fastest path from Python class to usable UI; not a production constraint here |
| Environment | `python-dotenv` | `.env` excluded from git; API key never touches source control |

---

## Setup

**Prerequisites:** Python 3.9+, an OpenAI API key.

```bash
# 1. Clone and enter
git clone <repo-url>
cd study-agent-HPE

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
echo "OPENAI_API_KEY=sk-..." > .env

# 4. Drop PDFs into the right folders
# module_materials/hpe_micro/lectures/
# module_materials/hpe_micro/readings/
# module_materials/hpe_micro/transcripts/
# (same pattern for macro, hpe_history, hpe_UP1, hpe_UP2)

# 5. Ingest a module
python ingest.py --module micro       # one module
python ingest.py --module all         # all five

# 6. Run the app
streamlit run app.py
```

To verify ingestion before running the full app:
```bash
python test_load.py          # confirms PyPDFLoader + chunking
python test_retrieval.py     # confirms ChromaDB retrieval
python test_chain.py         # confirms end-to-end chain + prompt comparison
```

To query from the CLI without the UI:
```bash
python agent.py --module micro --mode study_guide --question "market structures"
python agent.py --module history --mode book_compare --question "Mazower vs Swanson"
```

---

## Technical Decisions

### Why separate ChromaDB collections per module?

The alternative — a single collection with module as a metadata filter — is simpler to manage but conflates retrieval across disciplines. A question about "supply" should retrieve microeconomics chunks, not a Mazower passage that happens to mention food supply in interwar Yugoslavia. Separate collections make the retrieval space semantically coherent by design, not by filter.

The trade-off is operational: five collections means five embedding jobs and five persist directories. For a single-user local app, that cost is zero. If this scaled to a multi-user platform with hundreds of modules, a metadata-filtered single collection (or namespace-scoped Pinecone index) would be the right call.

### Why different chunk sizes for history vs economics?

Economics lectures are dense with definitions, equations, and short examples. A 500-character chunk captures a concept cleanly without bleeding into adjacent ones. Overlap of 50 characters handles edge cases at paragraph breaks.

History chapters are built around sustained arguments — a claim developed across three paragraphs before the evidence lands. Truncating at 500 characters often cuts the argument from its support. Testing with Mazower and Schorske showed that 800-character chunks preserved enough argumentative context for the LLM to answer historiographical questions coherently, rather than retrieving disconnected fragments that sounded relevant but couldn't be synthesised.

This is a real heuristic, not a universal truth. Ideal chunk size is a function of the writing style of the source material, not the subject. A densely-argued economics paper would warrant larger chunks; a bullet-point history summary would not.

### Why k varies by mode?

Simple Q&A needs 5 focused chunks. Cross-book modes like `theme_mapper` and `example_bank` need 15 to pull evidence from all four books. Retrieving too many chunks inflates the context window, increases cost, and risks the LLM averaging across conflicting sources rather than surfacing the strongest ones. The k values were tuned empirically by reading outputs, not set arbitrarily.

### Why temperature 0.3?

Study content should be reproducible. A student running `flashcards` twice on the same topic should get the same cards. Temperature 0.3 allows enough variation for natural language output without introducing answer drift. Essay planning modes could arguably go higher to encourage creative angles — that's a future experiment.

### Why `RetrievalQA` over a custom retrieval loop?

`RetrievalQA` is a thin wrapper. The retriever, prompt, and LLM are all injected — there is no hidden behaviour. Using it keeps the code readable without sacrificing control. If I needed multi-hop retrieval, query rewriting, or conversational memory, I would replace it with a LangGraph agent or a raw retrieval loop.

---

## Features

| # | Mode | Modules | k | Description |
|---|---|---|---|---|
| 1 | `qa` | All | 5 | Direct answer grounded in retrieved context with inline source citations |
| 2 | `study_guide` | All | 8 | Structured guide: overview, key concepts, core arguments, names/dates, connections, exam angles |
| 3 | `exam_questions` | All | 8 | 3 short-answer + 2 essay questions; past paper chunks weighted for command word patterns |
| 4 | `flashcards` | All | 5 | 10 Anki-style Q&A pairs focused on definitions, thinkers, cause-effect |
| 5 | `explain` | All | 5 | Three-layer explanation: plain English → academic → concrete example from material |
| 6 | `essay_plan` | All | 5 | Timed exam plan: thesis, 3 body paragraphs with evidence, counter-argument, conclusion |
| 7 | `definition_bank` | All | 5 | 8 entries formatted as thinker/concept: definition: key argument |
| 8 | `priority_score` | All | 8 | HIGH/MEDIUM/LOW priority assessment based on frequency in course material vs past papers |
| 9 | `equation_practice` | Micro, Macro | 8 | 3 worked numerical problems: setup, equation, worked solution, harder variant |
| 10 | `book_themes` | History | 8 | Historiographical argument, 3–5 major themes, 3 essay questions each with thesis starter |
| 11 | `chapter_summary` | History | 8 | Main argument, key examples used, historiographical position of a specific chapter |
| 12 | `book_compare` | History | 15 | Side-by-side: arguments, methodologies, periods covered, agreements, tensions, synthesis |
| 13 | `essay_practice` | History | 8 | Two-step: plan on first call; feedback + paragraph rewrite on `question \|\|\| student_answer` |
| 14 | `example_bank` | History | 15 | All named events, cases, and figures across all four books for a given theme |
| 15 | `theme_mapper` | History | 15 | Top 5 cross-book themes with per-book treatment, agreements, divergences, and essay sentences |

---

## Cost Estimates

All figures assume `text-embedding-3-small` ($0.02/1M tokens) and `gpt-4o-mini` ($0.15/1M input, $0.60/1M output).

| Operation | Approx. tokens | Approx. cost |
|---|---|---|
| Ingest one micro lecture PDF (~80 pages) | ~40k tokens | ~$0.0008 |
| Ingest all 5 modules from scratch | ~300k tokens | ~$0.006 |
| Single query (k=5, avg mode) | ~2k tokens in, ~400 out | ~$0.0005 |
| Single query (k=15, cross-book mode) | ~5k tokens in, ~600 out | ~$0.0011 |
| 50 queries/day across revision period (3 weeks) | ~210k tokens in, ~42k out | ~$0.057/day |
| Full revision period (3 weeks, 50 queries/day) | — | **~$1.20 total** |

The dominant cost is time, not money. At scale (multi-user, nightly re-ingestion of new uploads), the ingestion pipeline would be the place to optimise — batching embedding calls and caching chunk hashes to skip already-indexed documents.

---

## What I Would Add Next

**Conversational memory.** The current architecture is stateless — each query is independent. Adding a short-term message history (last 3–5 turns) via `ConversationBufferWindowMemory` would let the agent follow up on its own answers, which matters most for `essay_practice` and `explain` modes where the student wants to dig deeper.

**Query rewriting.** Short, ambiguous queries like "supply curve shift" produce mediocre retrieval. A rewriting step — asking the LLM to expand the query before embedding it — consistently improves recall for vague questions. HyDE (Hypothetical Document Embeddings) is worth testing for the history modes specifically.

**Chunk-level source verification.** Currently source citations are file-level. A better UX would surface the exact passage retrieved, highlight it, and let the student navigate to it — closer to how Elicit or Consensus work. This requires storing page numbers and character offsets at ingestion time, which `PyPDFLoader` already provides in metadata.

**Re-ranking.** Adding a cross-encoder re-ranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) between retrieval and generation would improve precision, especially for the k=15 cross-book modes where some retrieved chunks are topically adjacent but not directly relevant.

**Evaluation harness.** There is no systematic way to know if a prompt change improves answers. The right fix is a small golden dataset — 20–30 question/answer pairs per module, manually verified against the source material — and a script that scores faithfulness (answer grounded in context) and relevance (context retrieved for the right reasons) using an LLM judge. Without this, prompt iteration is just intuition.

**Authentication + multi-user support.** The current app is single-user by design. Moving to a hosted version would require per-user collections (or namespace isolation), a proper secret store instead of `.env`, and rate limiting on the embedding endpoint.

---

*Built during exam term, April 2026.*

## NOTE
Course materials (PDFs) are excluded from this repository and must be added locally. Run python ingest.py --module [module] after adding your own PDFs to the relevant folders.
