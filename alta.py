#Nicolae Giulia Gabriela
#Facultatea de Administratie si Afaceri, Universitatea din Bucuresti
#Cibernetica, Statistica si Informatica Economica, anul III

import pandas as pd
import re # regular expressions 

# Crearea variabilei df care citeste fisierul excel
df = pd.read_excel('finfact.xlsx')
print(f"\n Afisarea setului de date initial: \n {df.head(7)} ") 

# df2 = pd.read_ urmeaza sa prelucrez RFC ---------------------------------------------


# Pregatirea pentru curatare 
def pregatire_dataframe(df):
    coloane_de_sters=['url','source','posted','author','image_data','visualization_bias']
    df= df[df['visualization_bias'].astype(str) !='1'] # stergere bias-uri vizuale deoarece nu pot fi procesate de modelul NLP
                                                       #am transformat in string-uri pentru a asigura o comparatie corecta cu 1 pentru fiecare celula

    df = df.drop(columns=coloane_de_sters) # eliminarea coloanelor irelevante analizei
    df = df.dropna(subset=['claim', 'label']) # eliminare randuri cu valori nule din coloanele esentiale pentru analiza
    print(f"Numarul de duplicate gasite este: {df.duplicated(subset=['claim']).sum()}") #verific existenta claim-urilor duplicate
    df = df.drop_duplicates(subset=['claim'])
    print(f"Numarul de randuri ramase este: {len(df)}")
    return df

df = pregatire_dataframe(df)
print(df.head(7))

def proceseaza_text(text_brut):
    if not isinstance(text_brut, str): 
        return ""
     
    if "'sentence':" in text:
        gasit = re.findall(r"'sentence':\s*\"(.*?)\"|'sentence':\s*'(.*?)'", text) # Extrag textul din 'sentence': "..." sau 'sentence': '...' folosind regex, pt a elimina structura de tip dictionar ( fiind un fisier initial extras de tip json)
        text = " ; ".join([s[0] if s[0] else s[1] for s in gasit])                 # Daca exista mai multe fraze, le lipesc cu punct si virgula intre ele
        
    text = text.replace('\\ ;', ';').replace('\\', '')   # eliminare backslash-uri care pot aparea in urma eportului datelor
    text = re.sub(r'https?://\S+', '', text)             # eliminare URL-uri
    text = re.sub(r'ISBN\s?[\d-]{10,17}', '', text)      # eliminare coduri ISBN
    text = re.sub(r'\(pp?\.?[\d,\s-]+\)', '', text)      # eliminare referinte la pagini
    text = re.sub(r'\(\d{4}\)', '', text)                # eliminare ani din paranteze 
    text = re.sub(r'[@#^&*]', '', text)                  # sterg simboluri speciale 
    text = re.sub(r'\s\.\s', ' ', text)                  # sterg punctuatia izolata ramasa dupa curatare
    text = re.sub(r'\s+', ' ', text).strip()             # inlocuire spatii multiple cu un singur spatiu
    text = re.sub(r'( ;)+', ' ;', text)                  # verific sa ramana un singur semn de punct si virgula intre fraze
    text = re.sub(r'(\s*\.\s*){2,}', '.', text)          # inlocuire secvente de puncte multiple cu un singur punct
    
    return text 

# Curatarea universala a seturilor de date - eliminare zgomot din celule
def curatare_universala(text_brut, lista_coloane_text):
    for col in lista_coloane_text:
        if col in df.columns:
            print(f"Se curata coloana: {col}")
            df[col] = df[col].apply(proceseaza_text)
            
    return df

lista_coloane_text = ['evidence', 'claim', 'sci_digest', 'justification']
df = curatare_universala(df, lista_coloane_text)



def lustruire_finala(text):
    if not isinstance(text, str): # daca textul nu e string, returnez un string gol pt evitarea eorilor de procesare mai tarziu
        return "" 
    text = re.sub(r'Screengrab/[^;?!\.]+', '', text) # eliminare referinte la screengrab care apar in justificari
    text = re.sub(r'(\s*\.\s*){2,}', '. ', text)     # inlocuire secvente de puncte multiple cu un singur punct
    text = re.sub(r'(?i)Sources\s?:?.*', '', text)   # eliminare tot care incep cu "Sources" urmate de orice text
    text = re.sub(r'Accessed\s+\d{1,2}\s+[a-zA-Z.]+\s+\d{4}', '', text) # eliminare referinte de tip data de accesare a surselor
    text = re.sub(r'\s+([.,;!?])', r'\1', text)      # eliminare spatii inaintea semnelor de punctuatie
    text = re.sub(r'^[.,;!?\s]+', '', text)          # eliminare semne de punctuatie sau spatii la inceputul textului
    text = re.sub(r'([,;!?])\s*\1+', r'\1', text)    # inlocuire secvente de semne de punctuatie repetate cu un singur semn
    text = re.sub(r'\s+', ' ', text).strip()         # inlocuire spatii multiple cu un singur spatiu si eliminare spatii la inceputul si sfarsitul textului
    text = text.replace('..', '.')                   # inlocuire fragmente de doua puncte cu un singur punct
    text = re.sub(r'[.,;!?]\s*[.,;!?]', '.', text)   # inlocuire secvente de oricare semne de punctuatie consecutive cu un singur punct
    if len(text) > 0:                                # daca celula nu e goala, transform prima litera in majuscula pentru aspect final text
        text = text[0].upper() + text[1:]
    return text

for col in ['sci_digest', 'justification',"evidence", "claim"]:
    if col in df.columns:
        df[col] = df[col].apply(lustruire_finala) 

print("Un exemplu de text lustruit: \n")
print(df['evidence'].iloc[3360])
print(df['justification'].iloc[3360])
print(df['claim'].iloc[3360])
print(df['sci_digest'].iloc[3360])
























curatare_universal( df, text_brut=None)

print(df.head(7))


# Aplic functia de curatare universala pe coloanele relevante cu text 
col_de_curatat = ['evidence', 'claim', 'sci_digest', 'justification']
coloane_curatate = 0
for col in col_de_curatat:
    if col in df.columns:
        print(f"Se curata coloana: {col}...")
        df[col] = df[col].apply(curatare_universala)
        coloane_curatate +=1
        
if coloane_curatate > 0:
    print(f" Au fost curatate {coloane_curatate} coloane")
else:
    print("Nu a fost gasita nicio coloana de curatat.")



#Algoritmul Sliding Window
# Media nr de cuvinte din justification - pentru a decide daca trebuie aplicat algoritmul sliding window
print(df['justification'].str.split().str.len().describe())

# Data augumentation - sliding window pentru justificari
def sliding_window(text, window_size=512, step=400):
    words = text.split()
    # daca textul e scurt, returneaza tot textul
    if len(words) <= window_size: 
        return [text]
    
    windows = []
    for i in range(0, len(words), step):
        # Luăm bucata de text
        window = ' '.join(words[i:i+window_size])
        windows.append(window)
        # Dacă am ajuns la finalul textului, ne oprim
        if i + window_size >= len(words):
            break
    return windows

#Aplicarea algoritmului Sliding Window pe justificari si construirea unui nou dataframe cu textele si etichetele rezultate
lista_text = [] #  lista pentru textele noi
lista_label = [] # lista pentru etichete

for i in range(len(df)): 
    # extrag textul si eticheta de pe randul curent
    text = str(df.iloc[i]['justification']) 
    label = df.iloc[i]['label'] 

    fragmente = sliding_window(text) # tai textul in bucati
    for fragment in fragmente: 
         # pun fragmentul curent in lista de texte si eticheta lui in lista de etichete
        lista_text.append(fragment)
        lista_label.append(label) 

# construire tabel nou din cele doua liste 
df_final = pd.DataFrame({ 'justification': lista_text, 'label': lista_label})
print(len(df_final)) # afisare nr total de randuri dupa procesare


# Creare Dictionar pentru label encoding
label_map = {
    'false': 0,
    'true': 1,
    'neutral': 2
}

# Label encoding
label_map = {'false': 0, 'true': 1, 'neutral': 2}
df['label_encoded'] = df['label'].map(label_map)
print(df[['label', 'label_encoded']].head(50))



df.to_csv('finfact_curatat.csv', index=False)
print("Setul de date curatat a fost salvat in finfact_curatat.csv")




curatare_universal(df)






"""
def main():
    # Codul de mai sus poate fi plasat într-o funcție main pentru a organiza mai bine codul și a permite rularea acestuia ca script.

if __name__ == "__main__":
    main()
    """
    
    