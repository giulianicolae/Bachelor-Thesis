import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import set_seed

torch.set_num_threads(os.cpu_count())
set_seed(42)

device = torch.device("cuda")
print(f"Antrenarea va rula pe {device}.")

if not os.path.exists("finfact_curatat.csv"):
    print("Fisierul nu a fost gasit.")
else:
    df = pd.read_csv("finfact_curatat.csv")
    print("Fisierul a fost gasit.")
    

# impartirea setului de date     
train_claims, val_claims, train_justs, val_justs, train_labels, val_labels = train_test_split(
    df['claim'].astype(str).tolist(),
    df['justification'].astype(str).tolist(),
    df['label_encoded'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label_encoded']
)


# clasa dataset foloseste Sentence Pairs pentru a folosi atat claim-ul, cat si justificarile in antrenarea modelului
# obiectul ia datele rand cu rand din excel si le tokenizeaza
class DatasetFinanciar(Dataset):
    # datele sunt salvate in memoria interna a obiectului  
    def __init__(self, claims, justifications, labels, tokenizer, max_len=512):
        self.claims = claims
        self.justifications = justifications
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.claims) #nr de randuri de date

    def __getitem__(self, item):
        encoding = self.tokenizer(
            text=self.claims[item],              
            text_pair=self.justifications[item],
            add_special_tokens=True, #tokeni speciali de dlimitare porpozitii
            max_length=self.max_len, # daca textul e mai scurt de 512 adauga spatii goale
            padding='max_length', 
            truncation=True,         #daca e mai lung de 512, il truncheaza
            return_attention_mask=True, 
            return_tensors='pt',     #returneaza tensori pytorch folositi de model
        )
        return {
            'input_ids': encoding['input_ids'].flatten(), #returneaza lista de numere rezultata care reprezinta cuv
            'attention_mask': encoding['attention_mask'].flatten(), #returneaza 1 sau 0, indica modelului daca textul e real sau spatiu gol
            # flatten transforma tensorul dintr-o matrice cu un rand intr-o lista simpla
            'labels': torch.tensor(self.labels[item], dtype=torch.long) 
        }
        
        

# model si tokenizer
model_name = 'ProsusAI/finbert'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.to(device)


# metrice de evaluare
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


# argumente pentru antrenare
training_args = TrainingArguments(
    output_dir='./rezultate_finbert',
    num_train_epochs=4,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=8,
    warmup_ratio=0.1,
    weight_decay=0.005,
    label_smoothing_factor=0.1,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    use_cpu=False,
    fp16=True,
    dataloader_num_workers=0,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=DatasetFinanciar(train_claims, train_justs, train_labels, tokenizer),
    eval_dataset=DatasetFinanciar(val_claims, val_justs, val_labels, tokenizer),
    compute_metrics=compute_metrics,
)


# rularea antrenarii
print(" Se incepe antrenarea modelului pe GPU.")

print(f"Device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Model device: {next(model.parameters()).device}")

trainer.train()
print(" Antrenarea a fost realizata cu succes.")


#raport final si evaluarea modelului
print ("-"*10)
print("\n Raport final de performanta: ")
print ("-"*10)

rezultate = trainer.evaluate()

print(f"Acuratete: {rezultate['eval_accuracy']:.4f}")
print(f"F1: {rezultate['eval_f1']:.4f}")
print(f"Precizie: {rezultate['eval_precision']:.4f}")
print(f"Recall: {rezultate['eval_recall']:.4f}")


# salvarea modelului
output_folder='./model_final_licenta'
model.save_pretrained(output_folder)
tokenizer.save_pretrained(output_folder)
print (f"Modelul a fost salvat in folderul: {output_folder}")





