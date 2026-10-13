import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import os

device = torch.device("cuda")
print(f"Testarea va rula pe {device}")

# incarcare model antrenat
model_path = './model_final_licenta'

if not os.path.exists(model_path):
    print(
        f"Eroare: Nu a fost gasit folderul cu modelul antrenat. ")
    exit()
print("Se incarca modelul optimizat FinBERT.")
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()  # activare mod de evaluare


# incarcare set df2
set_testare = "rfc_curatat.csv"

if not os.path.exists(set_testare):
    print(f"Eroare: Fisierul '{set_testare}' nu a fost gasit.")
    exit()

df2 = pd.read_csv(set_testare)
print(f"Fisierul a fost incarcat. Se proceseaza {len(df2)} stiri...")


# liste in care vor fi salvate predicțiile retelei neurale
pred_labels = []
confidence_scores = []


# rulare text rand cu rand
print("FinBERT analizeaza stirile financiare...")
with torch.no_grad():
    for idx, row in df2.iterrows():
        text_claim = str(row['claim'])

        # tokenizare
        inputs = tokenizer(
            text=text_claim,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
            truncation=True
        )

        # trimitere tensori pe placa video (GPU)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # obtinere predictie bruta a modelului
        outputs = model(**inputs)

        # transformare valori brute în brobabilitati standard între 0 si 1
        probabilities = F.softmax(outputs.logits, dim=-1).flatten().cpu().numpy()

        # alegere clasa cu probabilitatea cea mai mare (0, 1, 2)
        predicted_class = probabilities.argmax()
        confidence = probabilities[predicted_class]

        # salvarea rezultatelor in liste
        pred_labels.append(predicted_class)
        confidence_scores.append(confidence)

# salvare rezultate pt modelul econometric si adaugare noi coloane in df2
df2['pred_label'] = pred_labels
df2['confidence'] = confidence_scores

# salvare tabel
output_filename = "RFC_etichetat.csv"
df2.to_csv(output_filename, index=False)

print("\n" + "-" * 10)
print("Testarea a fost finalizata cu succes.")
print(f"Toate cele {len(df2)} stiri au primit etichete de la modelul FinBERT.")
print(f"Fisierul rezultat a fost salvat ca {output_filename}")
print("-" * 10)

print("Se va extrage un top 50 de stiri false, avand gradul cel mai ridicat de confidenta.")

df2['pred_label'] = df2['pred_label'].astype(int)
df_fake = df2[df2['pred_label'] == 0]

if len(df_fake) == 0:
    print("Modelul nu a etichetat nicio stire ca fiind falsa.")
else:
    df_fake_sorted = df_fake.sort_values(by=['confidence'], ascending=False) #sortare descrescator dupa scor de incredere
    top_50f = df_fake_sorted.head(50) #selectare primele 50 stiri

    #salvare separata a topului 50 stiri
    top50_file = "TOP50_FN.csv"
    top_50f.to_csv(top50_file, index=False)

    print(f"A fost salvat fisierul {top50_file}, continand primele {len(top_50f)} stiri.")
    

df_true = df2[df2['pred_label'] == 1]
top_50true = df_true.sort_values(by=['confidence'], ascending=False).head(50)
top_50true.to_csv("TOP50_ADEVARATE.csv", index=False)
print("A fost salvat fisierul TOP50_ADEVARATE.csv.")
