import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
import statsmodels.api as sm
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc
import os
fisier_date = "TOP100.csv"

if not os.path.exists(fisier_date):
    print("Fisierul nu a fost gasit.")
    exit()

print("Se incarca fisierul cu cele 100 de stiri.")
df = pd.read_csv(fisier_date)
df['posted'] = pd.to_datetime(df['posted']).dt.tz_localize(None).dt.normalize()
df['pred_label'] = pd.to_numeric(df['pred_label']) # Ne asiguram ca tipul stirii este numeric


# functia pentru calculul variabilei dependente (socul pe piata) bazata pe datele din yfinance
def extrage_fluctuatie_anormala(row):
    ticker = row['ticker']
    data_eveniment = row['posted']
    
    # creez fereastra de 45 de zile cu 30 de zile anterioare de tranzactionare sigure 
    data_start = (data_eveniment - timedelta(days=45)).strftime('%Y-%m-%d')
    data_end = (data_eveniment + timedelta(days=5)).strftime('%Y-%m-%d')
    
    try:
        # descarcare date istorice din yahoo finance
        istoric = yf.download(ticker, start=data_start, end=data_end, progress=False)
        
        if istoric.empty or len(istoric) < 10:
            return np.nan 
            
        # ma asigur ca indexul de la yfinance nu are timezone
        istoric.index = pd.to_datetime(istoric.index).tz_localize(None)
        
        #separare perioada istorica vs ziua evenimentului
        istoric['Fluctuatie'] = abs((istoric['Close'] - istoric['Open']) / istoric['Open'])
        
        istoric_anterior = istoric[istoric.index < data_eveniment].tail(30)
        ziua_evenimentului = istoric[istoric.index >= data_eveniment].head(1)
        
        if istoric_anterior.empty or ziua_evenimentului.empty:
            return np.nan
            
        # calculul mediei fluctuatiei si verificarea socului    
        media_volatilitate = istoric_anterior['Fluctuatie'].mean()
        vol_curenta = ziua_evenimentului['Fluctuatie'].values[0]
        
        # variabila dependenta y este binara: 1 daca avem fluctuatie anormala, 0 altfel
        if vol_curenta > media_volatilitate:
            return 1
        else:
            return 0
            
    except Exception as e:
        print(f"Eroare la ticker-ul {ticker}: {e}")
        return np.nan


#pregatirea datelor pentru regresie logistica
print("Se descarca datele financiare si se calculeaza socul pe piata.")
df['soc_piata'] = df.apply(extrage_fluctuatie_anormala, axis=1)

# eliminarea valorilor nule din variabila dependenta pentru a avea un set curat pentru regresie
df_regresie = df.dropna(subset=['soc_piata']).copy()
df_regresie['soc_piata'] = df_regresie['soc_piata'].astype(int)
print(f"Numarul de observatii ramase dupa curatare este {len(df_regresie)}")


# verificare distributie variabila dependenta
distributie = df_regresie['soc_piata'].value_counts()
print(f"Distributie Y (Soc Piata): \n{distributie}")

if len(distributie) < 2:
    print("Atentie, variabila dependenta are o singura clasa sau nicio clasa valida iar regresia nu poate rula.")
    exit()

# Regresia Logistica
# Y= soc_piata (variabila dependenta)
# X = pred_label (variabila independenta: 0 = Fake, 1 = True)
Y = df_regresie['soc_piata']
X = df_regresie['pred_label'] 
X = sm.add_constant(X) 

print("\n----- ESTIMARE REGRESIE LOGISTICA ------")
model_logit = sm.Logit(Y, X)
rezultat_logit = model_logit.fit(disp=False)
print(rezultat_logit.summary())

# calculul metricilor si pseudo-R2 (McFadden) 
llf = rezultat_logit.llf
llnull = rezultat_logit.llnull
pseudo_r2 = 1 - (llf / llnull)
print(f"McFadden Pseudo R-squared: {pseudo_r2:.4f}\n")

# diagnosticul modelului si testarea ipotezelor
print("\n---- DIAGNOSTIC MODEL ----")
# calcul predictii
df_regresie['pred_prob'] = rezultat_logit.predict(X)
# clasificare binara pe baza pragului de 0.5
df_regresie['pred_class'] = np.where(df_regresie['pred_prob'] > 0.5, 1, 0)

# matricea de confuzie (goodness of fit)
cm = confusion_matrix(df_regresie['soc_piata'], df_regresie['pred_class'])
acc = accuracy_score(df_regresie['soc_piata'], df_regresie['pred_class'])
prec = precision_score(df_regresie['soc_piata'], df_regresie['pred_class'], zero_division=0)
rec = recall_score(df_regresie['soc_piata'], df_regresie['pred_class'], zero_division=0)

# metrici de performanta ale regresiei
print(f"Matrice de confuzie:\n{cm}")
print(f"Acuratete: {acc:.4f}")
print(f"Precizie: {prec:.4f}")
print(f"Sensibilitate (Recall): {rec:.4f}")

# curba ROC si AUC
fpr, tpr, thresholds = roc_curve(df_regresie['soc_piata'], df_regresie['pred_prob'])
roc_auc = auc(fpr, tpr)
print(f"AUC (C-statistic): {roc_auc:.4f}")

plt.style.use('default')
sns.set_theme(style="whitegrid")

 # vizualizare matrice de confuzie
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Rata de False Positive')
plt.ylabel('Rata de True Positive')
plt.title('Curba ROC')
plt.legend(loc="lower right")
plt.savefig('Curba_ROC_Regresie.png', bbox_inches='tight')
plt.close()

# extragerea reziduurilor de tip Pearson
reziduuri_pearson = rezultat_logit.resid_pearson

# testul Shapiro-Wilk pentru normalitatea reziduurilor
stat_sw, p_sw = shapiro(reziduuri_pearson)
print(f"Shapiro-Wilk test: statistic={stat_sw:.4f}, p-value={p_sw:.4f}")

# exportarea rezultatelor 
tabel_rezultate = pd.DataFrame({
    'Coeficient': rezultat_logit.params,
    'Std_Error': rezultat_logit.bse,
    'Z_test': rezultat_logit.tvalues,
    'P_value': rezultat_logit.pvalues,
    'Exp(Coef)': np.exp(rezultat_logit.params)
})
tabel_rezultate.to_csv("Rezultate_Statistice_Regresie.csv", float_format='%.5f')
df_regresie.to_csv("BazaDate_Regresie_Finala.csv", index=False)

print("\nRezultatele regresiei logistice au fost salvate local.")