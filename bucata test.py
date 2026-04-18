# 1. Unificăm curățarea (Curățare Universală + Lustruire + Eliminare Sources/Screengrab)
def curatare_totala(text_brut):
    text = str(text_brut)
    
    # Eliminare structură JSON/Dictionary
    if "'sentence':" in text:
        gasit = re.findall(r"'sentence':\s*\"(.*?)\"|'sentence':\s*'(.*?)'", text)
        text = " ; ".join([s[0] if s[0] else s[1] for s in gasit])
    
    # Eliminare URL, ISBN, Referințe, Sources, Screengrab, Ani
    text = re.sub(r'https?://\S+|ISBN\s?[\d-]{10,17}|\(pp?\.?[\d,\s-]+\)|\(\d{4}\)', '', text)
    text = re.sub(r'(?i)Sources\s?:?.*|Screengrab/[^;?!\.]+|Accessed\s+\d{1,2}\s+[a-zA-Z.]+\s+\d{4}', '', text)
    text = re.sub(r'[@#^&*]', '', text)
    
    # Lustruire finală punctuație
    text = re.sub(r'\s+([.,;!?])', r'\1', text)
    text = re.sub(r'^[.,;!?\s]+', '', text)
    text = re.sub(r'(\s*\.\s*){2,}|([,;!?])\s*\1+|[.,;!?]\s*[.,;!?]', '.', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) > 0:
        text = text[0].upper() + text[1:]
    return text

# 2. Aplicăm curățarea pe TOATE coloanele relevante înainte de orice altceva
for col in ['evidence', 'claim', 'sci_digest', 'justification']:
    if col in df.columns:
        df[col] = df[col].apply(curatare_totala)

# 3. Facem Label Encoding acum (ca să se propage la Sliding Window)
label_map = {'false': 0, 'true': 1, 'neutral': 2}
df['label_encoded'] = df['label'].map(label_map)

# 4. Aplicăm Sliding Window eficient (folosind .explode)
# Această metodă multiplică automat rândurile păstrând TOATE coloanele (label, claim, evidence)
df['justification'] = df['justification'].apply(lambda x: sliding_window(str(x), window_size=400, step=300))
df = df.explode('justification').reset_index(drop=True)

# 5. Salvarea finală
df.to_csv('finfact_curatat.csv', index=False)