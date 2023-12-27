from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus


# max_features dans mon vectorizer : garder les 5000 mots les plus fréquents
# max_df enlevé
# récupérer vectorizer pour récupérer score qui est la likelyhood à croiser avec la perplexity
def create_bow(corpus):
    """ Créer une représentation Bag of Words du corpus. """
    # Création de l'objet CountVectorizer avec une expression régulière pour exclure les nombres 
    vectorizer = CountVectorizer(max_features=5000)
    sparse_matrix_bow = vectorizer.fit_transform(corpus.apply(" ".join))
    dictionary_gensim = Dictionary([doc.split() for doc in corpus.apply(" ".join)])
    gensim_corpus_bow = Sparse2Corpus(sparse_matrix_bow, documents_columns=False)
    
    return sparse_matrix_bow, vectorizer, gensim_corpus_bow, dictionary_gensim

def create_tfidf(corpus):
    """ Créer une représentation TF-IDF du corpus. """
    vectorizer = TfidfVectorizer(token_pattern='(?u)\\b[a-zA-Z]+\\b', max_df=0.95)
    sparse_matrix_tfidf = vectorizer.fit_transform(corpus.apply(" ".join))
    dictionary_gensim = Dictionary([doc.split() for doc in corpus.apply(" ".join)])
    gensim_corpus_tfidf = Sparse2Corpus(sparse_matrix_tfidf, documents_columns=False)   
        
    return sparse_matrix_tfidf, vectorizer.get_feature_names_out(), gensim_corpus_tfidf, dictionary_gensim
