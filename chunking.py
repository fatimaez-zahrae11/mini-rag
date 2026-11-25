from pypdf import PdfReader
import re 
import json 

CHUNK_SIZE = 500
OVERLAP = 100

def load_pages(path):
    reader = PdfReader(path)
    return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]

def chunk_text(text, size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

pages = load_pages("chapitre.pdf")

chunks = []
for page_num, text in pages:
    text = " ".join(text.split())   # nettoie retours à la ligne et espaces multiples
    # supprime le pied de page répété sur chaque page
    text = re.sub(
        r"Chapter 13 [A-Za-z ,/-]{0,80}Database Administrator.s Guide E96348-21 Copyright.{0,120}?Page \d+ of \d+",
        "", text)
    for c in chunk_text(text, CHUNK_SIZE, OVERLAP):
        chunks.append({"page": page_num, "text": c})

print("Nombre de chunks :", len(chunks))
print("Longueur moyenne :", sum(len(c["text"]) for c in chunks) // len(chunks))
for c in chunks[10:13]:
    print("\n--- page", c["page"], "---")
    print(c["text"])


# transformer les 318 chunks en vecteurs :

with open("chunks.json", "w", encoding="utf-8") as f :
    json.dump(chunks, f, ensure_ascii=False)
print("chunks.json sauvegarde")