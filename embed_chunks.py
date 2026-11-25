import json 
from sentence_transformers import SentenceTransformer 

chunks = json.load(open("chunks.json", encoding="utf-8"))
model= SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

texts = [c["text"] for c in chunks]
vectors = model.encode(texts, show_progress_bar=True)

print("Nombre de vecteurs :", len(vectors))
print("Taille d'un vecteur :", len(vectors[0]))
print("5 premières valeurs du chunk 0 :", vectors[0][:5])