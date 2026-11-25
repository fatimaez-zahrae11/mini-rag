from sentence_transformers import SentenceTransformer, util


# PART A : voir la similarite de sens 
"""le modele paraphrase-multilingual-MiniLM-v2 transforme n'importe en 384 nombres 
    2 textes de sens proches donnent des vecteurs proches , qlq soit la langue 
"""
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

question = "Comment agrandir l'espace de stockage d'une base de donnees ?"


phrases = [
    "Use ALTER TABLESPACE to add a datafile",   # même sens, en anglais
    "Add more disk space to the database",      # même sens, autres mots
    "Create a new user and grant privileges",   # sujet différent
    "The weather is nice today",                # sans rapport
]

q_vec =model.encode(question)
p_vecs=model.encode(phrases)

# on mesure cette proximite de sens avec la similarite cosinus : proche de 1 = meme sens 
# proche de 0 = sans rapport 
scores =util.cos_sim(q_vec,p_vecs)[0]

for phrase,score in zip(phrases, scores) :
    print(f"{score:.2f}  {phrase}")     