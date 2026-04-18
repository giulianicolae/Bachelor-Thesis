
import pandas as pd
import re # regular expressions 

# Crearea variabilei df care citeste fisierul excel
df1 = pd.read_excel('finfact.xlsx')
print(f"\n Afisarea setului de date initial: \n {df1.head(7)} ") 

# df2 = pd.read_ urmeaza sa prelucrez RFC ---------------------------------------------

lista_coloane_df1 = ['claim', 'sci_digest', 'justification', 'issues', 'evidence', 'label']
lista_coloane_text_df1 = ['evidence', 'claim', 'sci_digest', 'justification', 'issues']



# Pregatirea pentru curatare 
def pregatire_dataframe(df):
    coloane_de_sters=['url','source','posted','author','image_data','visualization_bias']
    df= df[df['visualization_bias'].astype(str) !='1'] # stergere bias-uri vizuale deoarece nu pot fi procesate de modelul NLP
                                                       #am transformat in string-uri pentru a asigura o comparatie corecta cu 1 pentru fiecare celula
    df = df.drop(columns=coloane_de_sters) # eliminarea coloanelor irelevante analizei
    df = df.dropna(subset=['claim', 'label', 'justification']) # eliminare randuri cu valori nule din coloanele esentiale pentru analiza
    print(f"Numarul de duplicate gasite este: {df.duplicated(subset=['claim']).sum()}") #verific existenta claim-urilor duplicate
    df = df.drop_duplicates(subset=['claim'])
    print(f"Numarul de randuri ramase este: {len(df)}")
    return df

df1 = pregatire_dataframe(df1)
print(df1.head(7))



def procesare_text(text):
    if not isinstance(text, str): 
        return ""
     
    if "'sentence':" in text:
        gasit = re.findall(r"'sentence':\s*\"(.*?)\"|'sentence':\s*'(.*?)'", text) # Extrag textul din 'sentence': "..." sau 'sentence': '...' folosind regex, pt a elimina structura de tip dictionar ( fiind un fisier initial extras de tip json)
        text = " ; ".join([s[0] if s[0] else s[1] for s in gasit])                 # Daca exista mai multe fraze, le lipesc cu punct si virgula intre ele
        
    #sterg blocuri de text
    text = re.sub(r'https?://\S+', '', text)             # eliminare URL-uri
    text = re.sub(r'ISBN\s?[\d-]{10,17}', '', text)      # eliminare coduri ISBN
    text = re.sub(r'Accessed\s+\d{1,2}\s+[a-zA-Z.]+\s+\d{4}', '', text) # eliminare referinte de tip data de accesare a surselor
    text = re.sub(r'(?i)Sources\s?:?.*', '', text)   # eliminare tot care incep cu "Sources" urmate de orice text
    text = re.sub(r'Screengrab/[^;?!\.]+', '', text) # eliminare referinte la screengrab care apar in justificari
    text = re.sub(r'(?i)Sources\s?:?.*', '', text) # sterg "sources"
    text = re.sub(r'\(pp?\.?[\d,\s-]+\)', '', text)      # eliminare referinte la pagini
    text = re.sub(r'\(\d{4}\)', '', text)                # eliminare ani din paranteze 
    regex_date = r'\b(?:\d{1,2}\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b'
    text = re.sub(regex_date, '', text)
    
    #sterg punctuatia in plus
    text = text.replace('\\ ;', ';').replace('\\', '')   # eliminare backslash-uri care pot aparea in urma exportului datelor
    text = re.sub(r'[@#^&*]', '', text)                  # sterg simboluri speciale 
    text = re.sub(r'\s\.\s', ' ', text)                  # sterg punctuatia izolata ramasa dupa curatare
    
    #sterg cazurile de punctuatie ramasa dupa curatarea de zgomot
    text = re.sub(r'^[.,;!?\s]+', '', text)          # eliminare semne de punctuatie sau spatii la inceputul textului
    text = re.sub(r'^\[|\]$', '', text)              # eliminare paranteze drepte la inceputul sau sfarsitul textului
    text = re.sub(r'\s+([.,;!?])', r'\1', text)      # lipesc semnele de punctuatie de cuvantul dinaintea lor
    text = re.sub(r'( ;)+', ' ; ', text)                  # sa ramana un singur semn de punct si virgula intre fraze
    text = re.sub(r'([,;!?])\s*\1+', r'\1', text)    # inlocuire secvente de semne de punctuatie repetate cu un singur semn
    text = re.sub(r'(\s*\.\s*){2,}', '. ', text)          # inlocuire secvente de puncte multiple cu un singur punct
    text = re.sub(r'\s+', ' ', text).strip()             # inlocuire spatii multiple cu un singur spatiu si eliminare spatii la inceputul si sfarsitul textului
    
    if len(text) > 0:                    # transform prima litera in majuscula pentru aspect final text, daca celula nu e goala 
        text = text[0].upper() + text[1:]
    return text



# Curatarea universala a seturilor de date - eliminare zgomot din celule
def curatare_universala(df, lista_coloane_text_df1):
    for col in lista_coloane_text_df1:
        if col in df.columns:
            print(f"Se curata coloana: {col}")
            df[col] = df[col].apply(procesare_text)
            
    return df


df1 = curatare_universala(curatare_universala(df1, lista_coloane_text_df1), lista_coloane_text_df1)
print(df1.at[20, 'justification'])


def analiza_statistica(df1, coloana='justification'):
    print(f"\n Statistica lungimii cuvintelor pentru {coloana} ")
    print(df1[coloana].str.split().str.len().describe())

analiza_statistica(df1, 'justification')

# ALGORITMUL SLIDING WINDOW
# Data augumentation - sliding window pentru justificari
def sliding_window(text, window_size=400, step=300):
        words = str(text).split()
        if len(words) <= window_size: 
            return [text]
        windows = []
        for i in range(0, len(words), step):
            #iau bucata de text
            window = ' '.join(words[i:i+window_size])
            windows.append(window)
            if i + window_size >= len(words):
                break #daca ajung la finalul textului, ma opresc
        return windows

#Aplicarea algoritmului Sliding Window pe justificari si construirea unui nou dataframe cu textele si etichetele rezultate
def aplicare_sw(df, tinta='justification'):
    lista_randuri_noi = [] # Aici vom stoca randuri intregi, nu doar bucati de text

    for i in range(len(df)):
        date_rand = df.iloc[i].to_dict() # extrag randul curent sub forma de dictionar
        
        text_original = str(date_rand[tinta]) # iau textul pe care vreau sa aplic sliding window
        fragmente = sliding_window(text_original) # tai textul
        
        # pentru fiecare fragment, se creaza un rand nou care e o copie a celui original
        for fragment in fragmente:
            rand_nou = date_rand.copy()  # copiere a tuturor datelor din randul original
            rand_nou[tinta] = fragment # inlocuirea coloanei tinra cu fragmentul curent
            lista_randuri_noi.append(rand_nou)

    # transformare lista de dictionare intr-un dataframe nou
    df_rezultat = pd.DataFrame(lista_randuri_noi)
    
    print(f"Augmentare finalizata pentru coloana '{tinta}':")
    print(f"De la {len(df)} randuri la {len(df_rezultat)} randuri.")
    return df_rezultat

df1_final = aplicare_sw(df1, tinta='justification')


# Label encoding
def label_encoding(df):
    label_map = {'false': 0, 'true': 1, 'neutral': 2} # Creare Dictionar pentru label encoding
    df['label_encoded'] = df['label'].map(label_map)
    print(df[['label', 'label_encoded']].head(7))
    return df

df1_final = label_encoding(df1_final)

df1_final.to_csv('finfact_curatat.csv', index=False)
print("Setul de date curatat a fost salvat in finfact_curatat.csv")