import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

# Datensatz
file_path = 'data/complaints_processed.csv'

# Pandas - CSV in DataFrame einlesen
df = pd.read_csv(file_path)

# Zeilen mit fehlenden Beschwerden löschen
df.dropna(subset=['narrative'], inplace=True)

# Stopwords laden
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Funktion für „saubere“ Texte
def clean_text(text):
    text = text.lower() # Kleinschreibung vom Text
    text = re.sub(r'[^a-z\s]', '', text)  # Großbuchstaben, Zahlen, Satzzeichen und andere Sonderzeichen werden aus dem Text gelöscht
    text = ' '.join([word for word in text.split() if word not in stop_words]) # Vorverarbeitung der Daten mittels stopwords
    return text

# Funktion „saubere“ Texte aufrufen
df['cleaned_narrative'] = df['narrative'].apply(clean_text)

# Austesten
print(df.head(20))
print(stop_words)