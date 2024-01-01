from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# max_features dans mon vectorizer : garder les 5000 mots les plus fréquents
# max_df enlevé
# récupérer vectorizer pour récupérer score qui est la likelyhood à croiser avec la perplexity
def create_bow(corpus):
    """ Créer une représentation Bag of Words du corpus. """
    # Création de l'objet CountVectorizer avec une expression régulière pour exclure les nombres 
    vectorizer = CountVectorizer(max_features=5000)
    sparse_matrix_bow = vectorizer.fit_transform(corpus.apply(" ".join))
    
    return sparse_matrix_bow, vectorizer

def create_tfidf(corpus):
    """ Créer une représentation TF-IDF du corpus. """
    vectorizer = TfidfVectorizer(max_features=5000)
    sparse_matrix_tfidf = vectorizer.fit_transform(corpus.apply(" ".join))
        
    return sparse_matrix_tfidf, vectorizer
