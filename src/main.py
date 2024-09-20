import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from gensim import corpora
from gensim.models import CoherenceModel, LdaMulticore, Word2Vec
import time
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from sklearn.cluster import KMeans

def main():
    # Datensatz
    file_path = 'data/complaints_processed.csv'

    # Output Pfade
    output_path = 'output/'
    wordcloud_output_path = 'output/wordclouds/'

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

    # "saubere" Texte einlesen
    cleaned_texts = [text.split() for text in df['cleaned_narrative'] if text.strip()]
    
    # Austesten
    print(df.head(20))
    print(stop_words)
    print(cleaned_texts)

    # Datensätze für Gensim vorbereiten
    id2word = corpora.Dictionary(cleaned_texts)
    corpus = [id2word.doc2bow(text) for text in cleaned_texts]
    
    # Bereich der K-Werte für LDA definieren
    min_k = 2   # Wert kann adaptiert weden
    max_k = 5  # Wert kann adaptiert weden
    step_size = 1
    k_values = range(min_k, max_k + 1, step_size)
    coherence_scores = []
    model_list = []
    
    # LDA Parameter 
    passes = 10
    iterations = 50
    workers = 4
    
    # LDA-Modelle trainieren und berechnen von Coherence Score
    for k in k_values:
        print(f"Training des LDA-Modells für k={k} mithilfe von LdaMulticore.")
        start_time = time.time()
        lda_model = LdaMulticore(
            corpus=corpus,
            num_topics=k,
            id2word=id2word,
            passes=passes,
            iterations=iterations,
            random_state=42,
            workers=workers
        )
        end_time = time.time()
        print(f"Modell trainiert für k={k} in {end_time - start_time:.2f} Sekunden.")
        model_list.append(lda_model)
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=cleaned_texts,
            dictionary=id2word,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        coherence_scores.append(coherence_score)
        print(f'K: {k}, Coherence Score: {coherence_score}')

    # Den optimalen K-Wert finden
    optimal_k = k_values[coherence_scores.index(max(coherence_scores))]
    print(f'Der optimale Wert für K ist: {optimal_k}')
    
    # Coherence scores visualisieren
    plt.plot(k_values, coherence_scores)
    plt.xlabel('Anzahl Themen (K)')
    plt.ylabel('Coherence Score')
    plt.title('Coherence Score auf die Anzahl Themen')
    plt.savefig(os.path.join(output_path, 'coherence_scores.png'))
    plt.close()

    # Finales LDA-Modell trainieren basierend auf dem optimalen K
    print(f"Training des finalen LDA-Modells mit k={optimal_k}...")
    final_lda_model = LdaMulticore(
        corpus=corpus,
        num_topics=optimal_k,
        id2word=id2word,
        passes=passes,
        iterations=iterations,
        random_state=42,
        workers=workers
    )
    print("Endgültiges LDA-Modell trainiert.")
    

    # Funktion zum generieren von Wortwolken (LDA)
    def save_word_cloud_gensim(model, num_topics):
        for topic_idx in range(num_topics):
            top_words = [word for word, prob in model.show_topic(topic_idx, topn=100)]
            top_words_str = " ".join(top_words)
            wordcloud = WordCloud(width=1200, height=800, background_color='white').generate(top_words_str)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(f"Wortwolke für Thema {topic_idx}")
            output_file = os.path.join(wordcloud_output_path, f'lda_optimal_topic_{topic_idx}.png')
            plt.savefig(output_file)
            plt.close()
    
    # Funktion Wortwolken aufrufen
    save_word_cloud_gensim(final_lda_model, optimal_k)

    # Word2Vec-Modell trainieren
    print("Training des Word2Vec-Modells")
    word2vec_model = Word2Vec(sentences=cleaned_texts, vector_size=100, window=5, min_count=2, workers=workers)
    print("Word2Vec-Modell trainiert.")
    
    # Funktion zur Berechnung von Dokumentvektoren
    def document_vector(model, doc):
        doc = [word for word in doc if word in model.wv]
        if len(doc) == 0:
            return np.zeros(model.vector_size)
        return np.mean(model.wv[doc], axis=0)
    
    # Berechnung von Dokumentvektoren
    print("Berechnung von Dokumentenvektoren")
    word2vec_vectors = np.array([document_vector(word2vec_model, text) for text in cleaned_texts])
    
    # Dokumente mithilfe von KMeans clustern
    num_clusters = 5  # Wert kann adaptiert weden
    print(f"Clustering von Dokumenten in {num_clusters} Clustern mit KMeans.")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['word2vec_cluster'] = kmeans.fit_predict(word2vec_vectors)
    print("Clustering abgeschlossen.")
    
    # Funktion zum generieren von Wortwolken (Word2Vec-Cluster)
    for cluster in range(num_clusters):
        cluster_texts = ' '.join(df[df['word2vec_cluster'] == cluster]['cleaned_narrative'])
        wordcloud = WordCloud(width=1200, height=800, background_color="white").generate(cluster_texts)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Wortwolke für Word2Vec-Cluster {cluster}")
        output_file = os.path.join(wordcloud_output_path, f"word2vec_cluster_{cluster}.png")
        plt.savefig(output_file)
        plt.close()
    
    print("Wortwolken für Word2Vec-Cluster gespeichert.")
    print("Die Wortwolken sind im Ordner output/wordclouds einsehbar.")
    

if __name__ == '__main__':
    main()