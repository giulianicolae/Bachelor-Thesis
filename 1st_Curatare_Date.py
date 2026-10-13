# Curatarea seturilor de date FinFact si RFC si pregatirea pentru antrenarea modelului NLP 
import pandas as pd
import re # regular expressions  
from rapidfuzz import process, fuzz


# Crearea variabilelor df1 si df2
df1 = pd.read_excel('finfact.xlsx')
df2 = pd.read_excel('RFCdataset.xlsx')
print(f"\n Setul de date FinFact initial: \n {df1.head(10)} ") 
print(f" Setul de date RFC initial: \n {df2.head(10)} \n")


# Normalizarea variabilelor din RFC 
mapare_coloane_RFC ={
    'Ticker': 'ticker',
    'Date' : 'posted',
    'Link' : 'url',
    'Perturbed' : 'claim'
}
df2 = df2.rename(columns=mapare_coloane_RFC)
print(f"\n Setul de date RFC dupa redenumirea variabilelor: \n {df2.head(5)} ")


#Listele variabilelor vor fi folosite la curatarea seturilor de date
lista_coloane_df1 = ['claim','posted','sci_digest','justification','issues','evidence','label']
lista_coloane_text_df1 = [ 'claim', 'sci_digest', 'justification', 'issues','evidence']
lista_coloane_df2 = ['ticker', 'posted', 'url', 'claim']
lista_coloane_text_df2 = ['claim']


# Pregatirea pentru curatare 
def curatare_dataframe(df, nume_set):
    print (f"\n Se prelucreaza setul de date: {nume_set}")
    print(f'\n Numarul initial de randuri este: {len(df)}')
    
    #stergere randuri cu bias vizual
    if 'visualization_bias' in df.columns:
        df= df[df['visualization_bias'].astype(str) !='1']         
    
    #eliminare coloane irelevante analizei
    coloane_de_sters=['url','source','author','image_data','visualization_bias']
    df = df.drop(columns=coloane_de_sters, errors='ignore')
    
    #eliminare randuri cu valori nule in coloanele esentiale pentru analiza
    verific_valori_nule = ['claim', 'label', 'justification'] 
    for col in verific_valori_nule:
        if col in df.columns:
            df = df.dropna(subset=[col]) 
           
    df = df.drop_duplicates(subset=['claim'])
    return df

df1 = curatare_dataframe(df1 , 'FinFact')
df2 = curatare_dataframe(df2 , 'RFC')
print(f'\n Coloanele ramase dupa curatarea setului FinFact: \n {df1.head(7)}')
print(f'\n Coloanele ramase dupa curatarea setului RFC: \n {df2.head(7)}')
print('\n')


def elimina_duplicate_fuzzy(df, coloana='claim', prag=90):
    print(f"Se prelucreaza duplicatele fuzzy de pe coloana: {coloana}")
    initial_len = len(df)
    
    df = df.reset_index(drop=True)
    indice_de_sters = set()
    claims = df[coloana].tolist()
    
    for i in range(len(claims)):
        if i in indice_de_sters:
            continue

        matches = process.extract(claims[i], claims[i+1:], scorer=fuzz.token_sort_ratio, score_cutoff=prag)
        
        for match in matches:
            real_index = i + 1 + match[2]
            indice_de_sters.add(real_index)

    df = df.drop(index=list(indice_de_sters))
    print(f"Au fost eliminate {initial_len - len(df)} duplicate semantice.")
    return df

df1 = elimina_duplicate_fuzzy(df1, prag=90)
df2 = elimina_duplicate_fuzzy(df2, prag=90)
print(f"Dimensiune finala FinFact dupa stergerea duplicatelor fuzzy: {len(df1)} ")
print(f"Dimensiune finala RFC dupa stergerea duplicatelor fuzzy: {len(df2)}")
print("\n")


#Curatarea continutului din coloanele de tip text 
def procesare_text(text):        
    if not isinstance(text, str): 
        return ""
    
    # elimin structura de tip json si extrag continutul aflat dupa 'sentence': "..." sau 'sentence': '...' folosind regex 
    if "'sentence':" in text: 
        gasit = re.findall(r"'sentence':\s*\"(.*?)\"|'sentence':\s*'(.*?)'", text) 
        text = " ; ".join([s[0] if s[0] else s[1] for s in gasit])    # daca exista mai multe fraze le lipesc cu punct si virgula intre ele
        
    #sterg blocuri de text
    text = re.sub(r'https?://\S+', '', text)             # stergere link-uri
    text = re.sub(r'ISBN\s?[\d-]{10,17}', '', text)      # eliminare coduri ISBN
    text = re.sub(r'Accessed\s+\d{1,2}\s+[a-zA-Z.]+\s+\d{4}', '', text) # eliminare referinte de accesare
    text = re.sub(r'Screengrab/[^;?!\.]+', '', text)     # eliminare referinte la screengrab care apar in justificari
    text = re.sub(r'(?i)Sources\s?:?.*', '', text)       # sterg "sources"
    text = re.sub(r'\(pp?\.?[\d,\s-]+\)', '', text)      # eliminare referinte la pagini
    text = re.sub(r'\(\d{4}\)', '', text)                # eliminare ani din paranteze 
    luni = r'Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?' #lista luni
    regex_date = (
    r'\b(?:\d{1,2}\s+)?(?:' + luni + r')\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b|' # Dec 29, 2022 sau 29 Dec 2022
    r'\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b|'                                                # 29/12/2022 sau 29.12.22
    r'\b\d{4}-\d{1,2}-\d{1,2}\b')                                                        # 2022-12-29 (Format ISO)
    text = re.sub(regex_date, '', text)                  # stergerea datelor indiferent de formatul in care apar
    
    #sterg punctuatia in plus
    text = text.replace('\\ ;', ';').replace('\\', '')   # eliminare backslash-uri care pot aparea in urma exportului datelor
    text = re.sub(r'[@#^&*]', '', text)                  # sterg simboluri speciale 
    
    #sterg cazurile de punctuatie ramasa dupa curatarea de zgomot
    text = re.sub(r'^[.,;!?\s]+', '', text)          # curat semne de punctuatie sau spatii la inceputul textului
    text = re.sub(r'^\[|\]$', '', text)              # sterg paranteze drepte la inceputul sau sfarsitul textului
    text = re.sub(r'\s+([.,;!?])', r'\1', text)      # lipesc semnele de punctuatie de cuvantul dinaintea lor
    text = re.sub(r'(\s*\.\s*){2,}', '. ', text)     # inlocuire secvente de puncte multiple cu un singur punct
    text = re.sub(r'( ;)+', ' ; ', text)             # sa ramana un singur semn de punct si virgula intre fraze 
    text = re.sub(r'\s+', ' ', text).strip()         # inlocuire spatii multiple cu un singur spatiu si eliminare spatii la inceputul si sfarsitul textului
    text = re.sub(r'[.,;!?]\s*([.,;!?])', r'\1', text) # daca am mai multe semne de punctuatie la finalul textului, pastrez doar ultimul semn
    text = re.sub(r'([,;!?])\s*\1+', r'\1', text)    # inlocuire secvente de semne de punctuatie repetate cu un singur semn
    text = re.sub(r'[.,;:&*\s]+$', '', text)         # la sfarsitul textului inlocuiesc semnele de punctuatie irelevante cu punct
    text = re.sub(r'\s\.\s', ' ', text)              # sterg punctuatia izolata ramasa dupa curatare
    text = re.sub(r'""', '"', text)                  # corectare duplicate ghilimele
    if text:
        text += "."
    
    if len(text) > 0:                                # transform prima litera in majuscula pentru aspect final text, daca celula nu e goala 
        text = text[0].upper() + text[1:]
    return text



def curatare_rfc(df):
    
    if 'posted' in df.columns:
        df = df[df['posted'].astype(str).str.startswith("2025-")].copy()
    
    if 'claim' not in df.columns:
        return df
    
    # dictionar de traducere a secventelor corupte in caracterele corecte
    reparatii = {
        "â€™": "'",     
        "â€œ": '"',     
        "â€": '"',     
        "â€\x9d": '"',  
        "â€”": "—",     
        "â€“": "–",     
        "â€¦": "...",   
        "_x000D_": " ", # caracter Excel pentru Newline
        "\n": " ",      # newline-uri reale care rup textul
        "Â": "",
        "Ã©": "e"
    }
    
    for gresit, corect in reparatii.items():
        df['claim'] = df['claim'].astype(str).str.replace(gresit, corect, regex=False)
        
    print("Au fost corectate caracterele corupte din setul RFC.")
    return df

df2 = curatare_rfc(df2)


# Curatarea universala a seturilor de date - eliminare zgomot din celule
def curatare_universala(df, lista_coloane_text_df1):
    for col in lista_coloane_text_df1:
        if col in df.columns:
            print(f"Se curata coloana: {col}")
            df[col] = df[col].apply(procesare_text)
            
    return df

df1 = curatare_universala(df1, lista_coloane_text_df1)
df2 = curatare_universala(df2, lista_coloane_text_df2)


# Vizualizarea analizei statistice a lungimii cuvintelor din variabila justificari 
def analiza_statistica(df1, coloana, nume_set):
    print(f"\n Statistica lungimii cuvintelor pentru variabila {coloana} din setul de date {nume_set}:")
    print(df1[coloana].str.split().str.len().describe())

analiza_statistica(df1, 'justification' , 'FinFact')
analiza_statistica(df1, 'claim' , 'FinFact')
analiza_statistica(df2, 'claim' , 'RFC')


# Algoritmul Sliding Window
# Data Augumentation - sliding window pentru justificari
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
    
    print(f"\n Augumentare finalizata pentru coloana '{tinta}'.")
    print(f"De la {len(df)} randuri, la {len(df_rezultat)} randuri.")
    return df_rezultat

df1_final = aplicare_sw(df1, tinta='justification')



def lustruire_finala_justification(text):
    if not isinstance(text, str) or not text:
        return ""

    # sterg cuvintele de legatura dela finalul paragrafului care a fost truncat
    cuvinte_legatura = r'\b(?:and|but|or|tho|though|yet|with|so|for|if|then|because|although)\b\s*$'
    text = re.sub(cuvinte_legatura, '', text, flags=re.IGNORECASE).strip()

    # sterg semnele de punctuatie (de pauza) ramase la finalul sau la inceputul unui fragmet taiat
    text = re.sub(r'^[ :;,\-]+', '', text)
    text = re.sub(r'[ :;,\-]+$', '', text)

    # inlocuiesc secventele de doua semne de punctuatie doar cu al doilea semn
    while re.search(r'[.,;!?]\s*[.,;!?]', text):
        text = re.sub(r'[.,;!?]\s*([.,;!?])', r'\1', text)

    # sterg spatiul ramas la finalul textului
    text = text.rstrip()

    # prima litera devine majuscula
    if text:
        text = text[0].upper() + text[1:]
        
    return text

print("Lustruirea finala a coloanei justification a fost efectuata.")
df1_final['justification'] = df1_final['justification'].apply(lustruire_finala_justification)


# Label encoding
def label_encoding(df):
    label_map = {'false': 0, 'true': 1, 'neutral': 2} # Creare Dictionar pentru label encoding
    df['label_encoded'] = df['label'].map(label_map)
    print(df[['label', 'label_encoded']].head(7))
    return df

df1_final = label_encoding(df1_final)

# Fortez coloanele sa fie tipul corect 
df1_final['label_encoded'] = df1_final['label_encoded'].astype(int)
df1_final['claim'] = df1_final['claim'].astype(str)
df1_final['justification'] = df1_final['justification'].astype(str)
df1_final['posted'] = df1_final['posted'].astype(str)


print("\nSe salveaza seturile de date curatate in fisiere CSV.")
df1_final.to_csv('finfact_curatat.csv',index=False, encoding='utf-8-sig')
print("Setul de date FinFact curatat a fost salvat in finfact_curatat.csv.")
df2.to_csv('rfc_curatat.csv', index=False, encoding='utf-8-sig')
print("Setul de date RFC curatat a fost salvat in rfc_curatat.csv.")
print("Seturile de date curatate sunt gata pentru a fi folosite in etapa de antrenare a modelului NLP FinBert.\n")


#verificare finala a fistributiei
print("\nDistributia finala a stririlor in functie de label:")
print(df1_final['label'].value_counts())
print(df1_final['label'].value_counts(normalize=True) * 100) # Procente
