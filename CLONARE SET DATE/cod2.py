import pandas as pd
import json
import re

# 1. Load Data
file_path = r'C:/Users/GiuliaPC/Desktop/LICENTA/TOTUL DATE/CLONARE SET DATE/Fin-Fact/finfact.json'

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. Convert ID-keyed dictionary to a list
if isinstance(data, dict):
    clean_list = []
    for key, value in data.items():
        if isinstance(value, dict):
            value['doc_id'] = key
            clean_list.append(value)
    df = pd.json_normalize(clean_list)
else:
    df = pd.json_normalize(data)

# 3. Clean Text for Excel Compatibility
def clean_for_excel(text):
    if not isinstance(text, str):
        return text
    # Remove characters that break Excel (Control characters)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # If a cell starts with a formula trigger, add a leading space
    if text.startswith(('=', '+', '-', '@')):
        text = " " + text
    # Truncate if it exceeds Excel's absolute limit (rare, but safe)
    return text[:32760]

# Apply cleaning to all object/string columns
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(clean_for_excel)

# 4. Export using 'xlsxwriter' engine (more stable for large text)
try:
    with pd.ExcelWriter('finfact_cleaned.xlsx', engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    print("Success! The cleaned file is ready.")
except Exception as e:
    print(f"Export failed: {e}")