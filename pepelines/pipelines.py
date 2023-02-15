# importación de las bibliotecas necesarias
import spacy

nlp = spacy.load("en_core_web_sm")  # crear un objeto y cargar el modelo preentrenado para "Inglés"

text = '''Ravi and Raju are the best friends from school days.They wanted to go for a world tour and 
visit famous cities like Paris, London, Dubai, Rome etc and also they called their another friend Mohan to take part of this world tour.
They started their journey from Hyderabad and spent next 3 months travelling all the wonderful cities in the world and cherish a happy moments!
'''

# obtener los nombres propios del texto

doc = nlp(text)

pron_list = []

# buscar nombres propios
for token in doc:
    if token.pos_ == "PROPN":
        pron_list.append(token.text)


print(pron_list)


text = '''The Top 5 companies in USA are Tesla, Walmart, Amazon, Microsoft, Google and the top 5 companies in 
India are Infosys, Reliance, HDFC Bank, Hindustan Unilever and Bharti Airtel'''


doc = nlp(text)

# filtrar los nombres de las empresas

company_list = []

for ent in doc.ents:
    if ent.label_ == "ORG":
        company_list.append(ent.text)
