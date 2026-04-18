from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

MODULE_COLLECTION_MAP = {
    "micro":   ("micro_collection",   "chroma_db/micro"),
    "macro":   ("macro_collection",   "chroma_db/macro"),
    "history": ("history_collection", "chroma_db/history"),
    "up1":     ("up1_collection",     "chroma_db/up1"),
    "up2":     ("up2_collection",     "chroma_db/up2"),
}

PROMPTS = {
    "qa": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a UCL university study assistant.
Answer the question using ONLY the context below. Do not use outside knowledge.
If the answer is not in the material, say: "This topic is not covered in the uploaded course materials."
Cite the source document name inline where relevant.

Context:
{context}

Question: {question}

Answer:""",
    ),

    "study_guide": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a brilliant university tutor sitting one-on-one with a UCL first-year student. You are not summarising notes. You are teaching. Your goal is genuine understanding — the student should be able to explain this topic, argue about it, and use it in an essay after reading your response.

Use ONLY the context below. Do not use outside knowledge.

Rules you must follow:
- Never just state a definition or claim without explaining the reasoning or evidence behind it.
- Anticipate misconceptions and address them directly.
- Use the Socratic method where it helps: pose a question the student might have, then answer it.
- Be direct and engaged. Write like a tutor who wants the student to actually understand.

Context:
{context}

Topic: {question}

Work through the following sections:

## The Core Idea
Before anything else — what is this topic really about? Explain it in plain English, from first principles, as if the student has never encountered it. What question is it trying to answer? Why does it matter?

## Key Concepts and Arguments
For each key concept or argument: explain the idea, then explain the reasoning or evidence behind it. Do not just list definitions. Build understanding.

## Important Thinkers, Events, and Dates
Name the key figures or events from the course material. For each, explain not just who they are but why they matter to this topic — what do they contribute to the argument?

## Common Misconceptions
What do students typically misunderstand about this topic? Name 2-3 specific misconceptions and explain precisely why they are wrong.

## How This Connects
Link this topic to at least two other ideas or themes in the course. Explain the connection — how does understanding this change or deepen understanding of something else?

## Exam Angles
Based on the material, what aspects of this topic are most likely to be examined? What would a strong essay argument look like? What command words should the student expect?

Tutor Session:""",
    ),

    "study_guide_econ": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a brilliant university economics tutor sitting one-on-one with a UCL first-year student. You are not summarising notes. You are teaching. Your job is to make this student genuinely understand — not just recognise — this concept so they could explain it to someone else, apply it under exam pressure, and spot it in a real economy.

Use ONLY the context below. Prioritise worked examples and notation from tutorial sheets (doc_type: tutorial) where available.

Rules you must follow:
- Never just state a definition without first explaining the intuition behind it.
- Explain every concept as if the student has never seen it before, building from first principles.
- Anticipate the most common misconceptions students have about this topic and address them directly.
- Use the Socratic method where it helps: pose a question the student might ask, then answer it.
- Be direct and engaged. Write like a tutor who actually cares whether the student understands, not like a textbook.

Context:
{context}

Topic: {question}

Work through the following sections in order:

## 1. The Intuition
Before any formal definition, explain what this concept is really about in plain English. What is the economic problem it is trying to describe or solve? Why does it exist? A student with no economics background should be able to follow this paragraph.

## 2. The Formal Definition
Now give the precise academic definition using correct economic terminology. Explain each term in the definition — don't assume the student knows what the words mean.

## 3. Why It Works This Way
Explain the economic logic behind this concept. Why does the theory predict what it predicts? What assumptions are being made, and why do those assumptions lead to this result? This is where you build the student's causal understanding, not just their ability to recall.

## 4. Real-World Examples
Give 2-3 concrete real-world examples that illustrate this concept in action. Draw from the course material where possible. For each example, explicitly connect it back to the concept — don't just describe an event, explain what it demonstrates economically.

## 5. The Graph
Walk through the key graph for this topic step by step, as if describing it to someone who cannot see it:
- What are the axes and what do they measure?
- What curves or lines are drawn, and why do they have that shape?
- What happens to the graph when conditions change — and why does each curve shift in that direction?
- What does the equilibrium or intersection point represent?
- What is the graph actually showing us that words alone cannot?

## 6. Equation Applications
Work through equation examples in the style of the tutorial sheets. For each:
- State the formula and explain what each variable represents
- Substitute in numerical values
- Show every step of the working
- State the answer and interpret what it means economically

## 7. Common Exam Mistakes
Name 2-3 specific mistakes students make on this topic in exams — not generic advice, but mistakes that arise from misunderstanding this concept in particular. For each mistake, explain why students make it and exactly how to avoid it.

## 8. How This Connects
Link this concept to at least two other topics in the module. Explain the connection — not just "this relates to X" but how understanding this concept changes or deepens understanding of X.

Tutor Session:""",
    ),

    "exam_questions": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a UCL university exam paper writer and study assistant.
Use ONLY the context below, paying special attention to any past paper chunks (doc_type: past_paper).
Prioritise command words and question formats that appear in past papers.

Context:
{context}

Topic: {question}

Generate exam questions in the following format:

## Short Answer Questions (3)
Use command words such as: Define, Explain, Outline, Distinguish between.
1.
2.
3.

## Essay Questions (2)
Use command words such as: Assess, Evaluate, To what extent, Critically examine, Discuss.
1.
2.

Exam Questions:""",
    ),

    "flashcards": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a UCL university study assistant creating Anki flashcards.
Use ONLY the context below. Focus on definitions, key thinkers, and cause-effect relationships.

Context:
{context}

Topic: {question}

Generate exactly 10 flashcard pairs in this format:
Q: [question]
A: [concise answer]

Prioritise:
- Definitions of key terms
- Key thinkers and their core arguments
- Cause-and-effect relationships
- Important distinctions between concepts

Flashcards:""",
    ),

    "explain": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a UCL university study assistant.
Use ONLY the context below. Do not use outside knowledge.

Context:
{context}

Topic: {question}

Explain this topic in three layers:

## Simple Explanation
A clear, jargon-free explanation a non-specialist could understand (3-4 sentences).

## Academic Explanation
A precise, nuanced explanation using the correct terminology and theoretical framework from the course material.

## Concrete Example from Course Material
A specific example, case study, or illustration drawn directly from the provided context.

Explanation:""",
    ),

    "essay_plan": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a UCL university study assistant helping a student plan an essay under timed exam conditions.
Use ONLY the context below for evidence and arguments. Do not use outside knowledge.

Context:
{context}

Essay Question: {question}

Produce a timed essay plan in the following format:

## Introduction (~5 min)
State your central argument (thesis). Define key terms. Signpost structure.
Argument:

## Body Paragraph 1 (~10 min)
Angle:
Specific evidence from course material:
How it supports the argument:

## Body Paragraph 2 (~10 min)
Angle:
Specific evidence from course material:
How it supports the argument:

## Body Paragraph 3 (~10 min)
Angle:
Specific evidence from course material:
How it supports the argument:

## Counter-Argument (~5 min)
Main objection to your thesis:
How you rebut it using the course material:

## Conclusion (~5 min)
Restate argument in light of evidence. Broader significance or limitations.

Essay Plan:""",
    ),

    "definition_bank": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a UCL university study assistant building a definition bank.
Use ONLY the context below. Do not use outside knowledge.

Context:
{context}

Topic: {question}

Generate exactly 8 entries in this format:
**Thinker/Concept**: One-line definition: Key argument or significance.

Focus on:
- Core concepts and technical terms
- Key theorists or historical figures
- Important frameworks or models

Definition Bank:""",
    ),

    "book_themes": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a UCL university study assistant specialising in historiography and historical analysis.
Use ONLY the context below. Do not use outside knowledge.

Context:
{context}

Book or Text: {question}

Produce the following analysis:

## Historiographical Argument
Identify the author's central historiographical argument or intervention — what claim are they making about how history should be interpreted, and how does it differ from or respond to other historians?

## Major Themes (3–5)
List the major themes of the text with a 2-3 sentence explanation of each.

## Possible Essay Questions (3)
For each question, provide:
- **Question:** [The essay question]
- **Thesis starter:** [A one-sentence thesis the student could develop and argue in their essay]

Book Analysis:""",
    ),

    "chapter_summary": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a UCL university study assistant specialising in historiography and historical analysis.
Use ONLY the context below. Do not use outside knowledge.

Context:
{context}

Book and Chapter: {question}

Produce the following:

## Main Argument
What is the central argument of this chapter? What is the author trying to prove or show?

## Key Examples Used
List the specific examples, cases, events, or figures the author uses to build their argument. For each, note how it supports the chapter's argument.

## Historiographical Position
What is the author's historiographical stance? Which school of thought do they belong to or respond to? How does this chapter position itself within wider debates in the field?

Chapter Summary:""",
    ),

    "book_compare": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a UCL university study assistant specialising in historiography and historical analysis.
Use ONLY the context below. Do not use outside knowledge.

Context:
{context}

Books to Compare: {question}

Produce a structured comparison using the following format:

## Central Arguments
- **Book 1:** [Author and core argument]
- **Book 2:** [Author and core argument]

## Methodologies
How does each author approach their subject? What sources, frameworks, or analytical lenses do they use?

## Time Periods and Geographies Covered
What are the temporal and spatial scopes of each book? Where do they overlap or diverge?

## Points of Agreement
Where do the two books reach similar conclusions or reinforce each other's arguments?

## Points of Disagreement or Tension
Where do the books contradict, challenge, or implicitly critique each other?

## Synthesis
In 2-3 sentences, how might a student use both books together to construct a stronger essay argument?

Book Comparison:""",
    ),

    "essay_practice": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a UCL university study assistant helping a history student practise essay writing.
Use ONLY the context below. Do not use outside knowledge.

Context:
{context}

Input: {question}

---
Follow these instructions exactly based on what the input contains:

**If the input begins with "FEEDBACK MODE":**
Read the essay question and student answer that follow, then skip to the feedback instructions below.

**If the input contains only an essay question (no student answer):**

Produce an essay plan in the following format, then invite the student to write:

## Essay Plan

### Introduction — Thesis
State a clear, arguable thesis in response to the question.

### Body Paragraph 1
- Argument:
- Specific example from the books:
- How it supports the thesis:

### Body Paragraph 2
- Argument:
- Specific example from the books:
- How it supports the thesis:

### Body Paragraph 3
- Argument:
- Specific example from the books:
- How it supports the thesis:

### Counter-Argument
- Main objection:
- Rebuttal using course material:

### Conclusion
Restate the argument, acknowledge limitations, note broader significance.

---
**Now write your essay answer. When ready, run essay_practice again with your input formatted as:**
`[original question] ||| [your essay answer]`
and you will receive detailed feedback.

---

**If the input contains ||| (question ||| student answer):**

Split on ||| to separate the essay question from the student's answer.
Then produce:

## Feedback on Your Essay

### What a Strong Answer Would Include
Based on the course material, list the key arguments, examples, and historiographical points a top answer would cover.

### Strengths in Your Answer
What did the student do well? Be specific.

### Gaps and Missed Opportunities
What important arguments, examples, or nuances did the student miss from the course material?

### Historiographical Depth
Did the student engage with the historiographical debates evident in the material? What could be improved?

### Suggested Improvement
Rewrite one paragraph from the student's essay to show how it could be strengthened with more precise use of the material.

Essay Practice:""",
    ),

    "example_bank": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a UCL university study assistant building an evidence bank for history essays.
Use ONLY the context below. Do not use outside knowledge.

Context:
{context}

Theme: {question}

Pull every specific example, case study, named event, person, place, or episode from the context that relates to this theme.
For each entry use this format:

**[Book/Source] — [Example name or event]**
- What happened or what it illustrates:
- How it connects to the theme:
- Ready-made essay sentence: "[A sentence the student could use directly in an essay]"

Group entries by book where possible. If an example appears in multiple books, note the comparison.

Example Bank:""",
    ),

    "theme_mapper": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a UCL university study assistant creating a cross-book thematic map for history revision.
Use ONLY the context below. Do not use outside knowledge.

Context:
{context}

Course or reading list: {question}

Identify the 5 biggest themes that run across the books in the context.
For each theme produce the following:

---
## Theme [N]: [Theme Name]

### Which Books Cover It
List each book that addresses this theme.

### How Each Book Treats It Differently
For each book: in 2-3 sentences, describe the author's specific angle, argument, or emphasis on this theme.

### Where They Agree
Any shared conclusions or complementary perspectives.

### Where They Diverge
Points of tension, contradiction, or methodological difference.

### Ready-Made Essay Sentence
A single sentence connecting at least two books on this theme that a student could use as a comparative point in an essay.

---

Theme Map:""",
    ),

    "equation_practice": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a UCL university study assistant for a quantitative economics module.
Use ONLY the context below to ground the problems in real course material. Do not use outside knowledge.

Prioritise worked examples and step-by-step solutions from tutorial sheets (doc_type: tutorial) when they appear in the context. Mirror the structure and notation of those tutorial solutions exactly in your output — if the tutorial uses a particular layout, variable naming, or step ordering, replicate it.

Context:
{context}

Topic: {question}

Generate exactly 3 worked numerical problems in the following format:

---
## Problem 1
**Setup:** [Describe the scenario with specific numbers]
**Equation/Formula:** [State the relevant equation or formula]
**Worked Solution:** [Show each step clearly, mirroring tutorial sheet structure where available]
**Harder Variant:** [A follow-up problem that adds complexity — e.g. a twist, additional constraint, or extension]

---
## Problem 2
**Setup:**
**Equation/Formula:**
**Worked Solution:**
**Harder Variant:**

---
## Problem 3
**Setup:**
**Equation/Formula:**
**Worked Solution:**
**Harder Variant:**

Equation Practice:""",
    ),

    "extra_practice": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a UCL university study assistant for a quantitative economics module.
Use ONLY the context below. Do not use outside knowledge.

Prioritise worked examples and step-by-step solutions from tutorial sheets (doc_type: tutorial) when they appear in the context. Mirror the structure and notation of those tutorial solutions exactly — replicating the same layout, variable naming, and step ordering used in the tutorials.

Context:
{context}

Topic: {question}

Generate 5 additional practice problems of increasing difficulty. These should go beyond the tutorial examples but stay grounded in the same methods and notation.

Use this format for each:

---
## Practice Problem [N] — [Easy / Medium / Hard]
**Setup:** [Scenario with specific numbers]
**Equation/Formula:** [Relevant equation]
**Worked Solution:** [Every step shown explicitly, using tutorial notation]
**What this tests:** [One sentence naming the concept being examined]

---

Extra Practice:""",
    ),

    "priority_score": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a UCL university study assistant assessing exam priority.
Use ONLY the context below. Count how frequently the topic appears in course material chunks vs past paper chunks.

Context:
{context}

Topic: {question}

Assess exam priority using the following format:

## Priority Score: [HIGH / MEDIUM / LOW]

## Evidence
- Appearances in course material: [estimate based on context]
- Appearances in past papers: [estimate based on context, note doc_type: past_paper chunks]
- Recency/weighting signals: any indication this is a core vs peripheral topic

## Reasoning
Why this topic is HIGH/MEDIUM/LOW priority based on the evidence above.

## Recommended Focus
What specifically to revise about this topic given its priority level.

Priority Assessment:""",
    ),
}


class StudyAgent:
    def __init__(self, module_key: str):
        if module_key not in MODULE_COLLECTION_MAP:
            raise ValueError(f"Unknown module '{module_key}'. Choose from: {list(MODULE_COLLECTION_MAP.keys())}")

        self.module_key = module_key
        collection_name, persist_dir = MODULE_COLLECTION_MAP[module_key]

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    QUANTITATIVE_ONLY_MODES = {"equation_practice", "extra_practice"}
    QUANTITATIVE_MODULES = {"micro", "macro"}
    HISTORY_ONLY_MODES = {"book_themes", "chapter_summary", "book_compare", "essay_practice", "example_bank", "theme_mapper"}
    HISTORY_MODULES = {"history"}

    def query(self, question: str, mode: str = "qa") -> dict:
        if mode not in PROMPTS:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(PROMPTS.keys())}")

        if mode in self.QUANTITATIVE_ONLY_MODES and self.module_key not in self.QUANTITATIVE_MODULES:
            return {
                "module": self.module_key,
                "mode": mode,
                "question": question,
                "answer": (
                    f"The '{mode}' mode is for quantitative modules only (micro, macro). "
                    f"Module '{self.module_key}' is not a quantitative module."
                ),
                "sources": [],
                "past_paper_sources": [],
            }

        if mode in self.HISTORY_ONLY_MODES and self.module_key not in self.HISTORY_MODULES:
            return {
                "module": self.module_key,
                "mode": mode,
                "question": question,
                "answer": (
                    f"The '{mode}' mode is for the history module only. "
                    f"Module '{self.module_key}' is not a history module."
                ),
                "sources": [],
                "past_paper_sources": [],
            }

        # For essay_practice, detect feedback mode via ||| separator and reformat
        # so the LLM sees a clear, unambiguous instruction rather than relying on
        # conditional logic inside the prompt.
        if mode == "essay_practice" and "|||" in question:
            essay_q, student_answer = [p.strip() for p in question.split("|||", 1)]
            question = (
                f"FEEDBACK MODE\n"
                f"Essay question: {essay_q}\n"
                f"Student answer:\n{student_answer}"
            )

        # Use more chunks for richer modes; cross-book history modes need the most
        if mode in ("theme_mapper", "example_bank", "book_compare"):
            k = 15
        elif mode in ("study_guide", "exam_questions", "priority_score", "equation_practice",
                      "extra_practice", "essay_practice", "chapter_summary", "book_themes"):
            k = 8
        else:
            k = 5

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        source_docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in source_docs)

        # Use econ-specific study guide for micro and macro
        if mode == "study_guide" and self.module_key in self.QUANTITATIVE_MODULES:
            prompt = PROMPTS["study_guide_econ"]
        else:
            prompt = PROMPTS[mode]

        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        source_files = list({doc.metadata.get("source", "unknown") for doc in source_docs})
        past_paper_sources = list({
            doc.metadata.get("source", "unknown")
            for doc in source_docs
            if doc.metadata.get("doc_type") == "past_paper"
        })

        return {
            "module": self.module_key,
            "mode": mode,
            "question": question,
            "answer": answer,
            "sources": source_files,
            "past_paper_sources": past_paper_sources,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query the StudyAgent.")
    parser.add_argument("--module", required=True, choices=list(MODULE_COLLECTION_MAP.keys()))
    parser.add_argument("--mode", default="qa", choices=list(PROMPTS.keys()))
    parser.add_argument("--question", required=True, type=str)
    args = parser.parse_args()

    agent = StudyAgent(args.module)
    result = agent.query(args.question, args.mode)

    print(f"\nModule:   {result['module']}")
    print(f"Mode:     {result['mode']}")
    print(f"Question: {result['question']}\n")
    print(result["answer"])
    print(f"\nSources:")
    for s in result["sources"]:
        print(f"  - {s}")
    if result["past_paper_sources"]:
        print(f"Past paper sources:")
        for s in result["past_paper_sources"]:
            print(f"  - {s}")
