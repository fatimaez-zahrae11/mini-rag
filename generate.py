import ollama
from retrieve import retrieve

MODEL = "llama3.2:3b"
DEBUG = True   # passe à False pour masquer les chunks

def build_prompt(question, results):
    context = "\n\n".join(
        f"[Page {meta['page']}] {doc}" for doc, meta, _ in results
    )
    return f"""Tu es un assistant qui répond à des questions sur le guide Oracle Database Administrator's Guide.
Règles :
- Réponds en français, uniquement avec les informations du CONTEXTE.
- N'invente rien : pas de syntaxe ni de détail absent du CONTEXTE.
- Si le CONTEXTE ne contient pas la réponse, réponds UNIQUEMENT : "L'information n'est pas dans le document."
- Si tu réponds, ne dis JAMAIS que l'information est absente.

CONTEXTE :
{context}

QUESTION : {question}"""

def answer(question, k=3):
    results = retrieve(question, k)

    if DEBUG:
        print("\n===== CHUNKS RETROUVÉS =====")
        for i, (doc, meta, dist) in enumerate(results, 1):
            print(f"\n#{i} | page {meta['page']} | similarité {1 - dist:.2f}")
            print(doc)
        print("\n===== RÉPONSE =====")

    r = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": build_prompt(question, results)}],
        options={"num_ctx": 2048, "temperature": 0},
    )
    pages = sorted({meta["page"] for _, meta, _ in results})
    return r["message"]["content"] + f"\n\nSources : pages {pages} du chapitre"

if __name__ == "__main__":
    while True:
        q = input("\nQuestion (vide pour quitter) : ")
        if not q.strip():
            break
        print(answer(q))