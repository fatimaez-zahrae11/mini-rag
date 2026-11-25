import json 
import chromadb 
from sentence_transformers import SentenceTransformer

""" pr l'instant , les 318 vecteurs vivent dans la memoire de embed_chunks.py et disparaissent a la fin du script .
    CromaDB es une BD specialisee : elle les stocke sur le disque et sait retrouver tres vite 
    les vecteurs les plus proches d'un vecteur donne 
"""

chunks = json.load(open("chunks.json",encoding="utf-8"))
model= SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# base sauvegardee sur le dissque dans le dossier chroma_db/

client = chromadb.PersistentClient(path="chroma_db")

# on repart de 0 si la collection existe deja ce qui evite es doublons 

try :
    client.delete_collection("chapitre13")
except Exception :
    pass 

collection = client.create_collection(
    name="chapitre13",
    metadata={"hnsw:space" : "cosine"},
)

texts= [c["text"] for c in chunks]

# PR CHAQUE CCHUNK ON  STOCKE 4 CHOSES : ids ( identifiant)
# documents ( le texte du chunk )
# embeddings ( le vecteur de 384 nombres )
# metadatas ( les meta-donnees de chaque chunk ex {"page":2})
collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=texts,
        embeddings=model.encode(texts, show_progress_bar=True).tolist(),
        metadatas=[{"page": c["page"]} for c in chunks],
)

""" note : dans une architecture RAG , on stocke le texte d'origine ( le chunk)
en plus de son vecteurs  parce qu'un vecteur ne sert qu'a chercher , alors 
que le LLM a besoin de texte lisible pr rediger sa reponse 
"""

print("Chunks stockés :", collection.count())
print(collection.get(ids=["0"], include=["documents", "metadatas"]))