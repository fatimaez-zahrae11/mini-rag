import chromadb
from sentence_transformers import SentenceTransformer 
model =SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection =client.get_collection("chapitre13")

def retrieve(question, K=3) :
    q_vec = model.encode([question]).tolist()
    res = collection.query(
        query_embeddings=q_vec,
        n_results=K,
        include=["documents", "metadatas", "distances"],
    )
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))

if __name__ == "__main__":
    while True:
        q = input("\nQuestion (vide pour quitter) : ")
        if not q.strip():
            break
        for rank, (doc, meta, dist) in enumerate(retrieve(q), 1):
            print(f"\n#{rank} | page {meta['page']} | similarité {1 - dist:.2f}")
            print(doc)



""" une petite precision pas de " je ne sais pas " - chromadb renvoie tjrs les K chunks les plus proches, m s'ils sont hors sujet 
    c'est une limite relle du RAG , q'on traite avec deux facons : 
    en disant au LLM de repondre " l'info nest pas presente dans le document
    ou avec un seuil de similarite ( on ignore les chunks sous par ex un seuil de 0.3)
"""
# la langue n'est pas un facteur genat ici : c l'avantage du modele multilingue 