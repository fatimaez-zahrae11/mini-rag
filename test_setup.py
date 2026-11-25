from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

reader = PdfReader("chapitre.pdf")
print("Pages :", len(reader.pages))
print("Extrait :", reader.pages[0].extract_text()[:200])

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
vec = model.encode("test")
print("Taille d'un embedding :", len(vec))

