# mini-rag

A small question-answering tool that reads one chapter of a PDF and answers questions about it. Everything runs locally and costs nothing.

This is my first mini project with RAG (Retrieval-Augmented Generation). I built it to understand how the pieces fit together, so the code is deliberately simple and split into small scripts, one per step of the pipeline.

The document I used is chapter 13 ("Managing Tablespaces") of the Oracle Database Administrator's Guide, about 50 pages. You can swap in any text-based PDF.

## Why RAG?

An LLMon its own answers from memory. It doesn't know your documents, and when it doesn't know something it can still produce an answer that sounds right(hallucinations). RAG fixes this by looking up the relevant passages first and handing them to the model, a bit like an open-book exam.

## Architecture

There are two phases. Indexing happens once, querying happens every time you ask something.

```
Indexing (run once)

  PDF --> chunks --> embeddings --> ChromaDB
          (500 chars,   (384 numbers   (vectors + text
           overlap 100)  per chunk)     + page number)


Querying (every question)

  question --> embedding --> ChromaDB --> top 3 chunks --> prompt --> LLM --> answer
               (same model)  similarity                    (rules +           + source
                             search                         context +          pages
                                                            question)
```

The two models have different jobs:

- **Embedding model** (`paraphrase-multilingual-MiniLM-L12-v2`): turns text into vectors so I can search by meaning instead of keywords. It's multilingual, so I can ask questions in French about an English document.
- **LLM** (`llama3.2:3b` through Ollama): writes the final answer using only the retrieved chunks.

## Stack

- Python 3.11
- `pypdf` to read the PDF
- `sentence-transformers` for embeddings
- `chromadb` as the vector database (cosine distance, saved on disk)
- `ollama` to run Llama 3.2 3B locally

## Project structure

```
mini-rag/
├── extract_chapter.py   cut a page range out of the big PDF
├── chunking.py          read, clean and split the text -> chunks.json
├── embeddings_demo.py   small demo of how similarity between sentences works
├── embed_chunks.py      sanity check: embed all chunks
├── index_chroma.py      store chunks + vectors in ChromaDB
├── retrieve.py          find the top-k chunks for a question
├── generate.py          build the prompt, call the LLM, print the answer
└── requirements.txt
```

## Running it

You need Python 3.11 and [Ollama](https://ollama.com) installed.

```bash
git clone <my repo>
cd mini-rag

python -m venv venv
# Windows (PowerShell):  venv\Scripts\activate
# macOS / Linux:         source venv/bin/activate

pip install -r requirements.txt
ollama pull llama3.2:/3b
```

Put your PDF in the project folder as `chapitre.pdf` (or use `extract_chapter.py` to cut a chapter out of a bigger one). The Oracle guide is under copyright, so it isn't in this repo; download it from Oracle's documentation site.

Then:

```bash
python chunking.py       # creates chunks.json
python index_chroma.py   # creates the chroma_db/ folder
python generate.py       # ask questions
```

The prompt in `generate.py` is written in French and asks the model to answer in French. Edit it if you want English answers.

## Settings I settled on

| Setting        | Value          | Why                                                                 |
| -------------- | -------------- | ------------------------------------------------------------------- |
| chunk size     | 500 characters | at 200, ideas got cut in half; at 500 most chunks hold a whole idea |
| overlap        | 100 characters | so a sentence on a chunk boundary isn't lost                        |
| top-k          | 3              | enough context without too much noise                               |
| temperature    | 0              | stable answers that stick to the context                            |
| context window | 2048 tokens    | my machine ran out of RAM with the default                          |

## What I learned

- **Use the same embedding model for chunks and questions.** Otherwise the vectors live in different spaces and comparing them means nothing.
- **ChromaDB never says "I don't know".** It always returns the k closest chunks(similarite minimale) , even if they're unrelated. The only protection is the prompt telling the LLM to say the information isn't in the document. When I asked about the capital of Morocco, it did exactly that.
- **When an answer is bad, look at the retrieved chunks.** If they contain the answer, the generation step is at fault (prompt, small model). If they don't, retrieval is (chunking, k, embeddings).
- semantic similarity between the query embedding and the document chunks is evaluated using cosing similarity

## Limitations

- Only one chapter is indexed.
- No similarity threshold, so an irrelevant chunk can still reach the LLM.
- I only tested a handful of questions by hand, there is no proper evaluation yet.
- Chunks are cut by character count, so words and sentences can be split.

## Ideas for next steps

- Add a similarity threshold
- test the pipline with diff parameters ( taille du chunk , overlap , top k chunks ......)
- Build Streamlit interface
- Index more chapters, then the whole guide
- Compare with a bigger LLM
