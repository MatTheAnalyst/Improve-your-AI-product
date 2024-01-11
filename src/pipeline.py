import pickle
import pandas as pd
from data_processing.general_preprocessing import clean_comment, tokenize_comment, delete_stopwords, lemmatized_tokens, delete_most_frequent_word

class PipelineLDA:
    def __init__(self, config:dict):
        self.lda_model = self.load_model(config['model_path'])
        self.vectorizer = self.load_vectorizer(config['vectorizer_path'])
        self.top_words = config.get('top_words', list())
        self.interpreted_topics = config.get('interpreted_topics', {})

# Dictionnaire de config. Fichier de config avec les path de chacun des outils. Sub config en YAML convertir une config -> dic à lire et à passer dans ma fonction.
# Ne pas recharger le modèle à chaque prédiction : charger le modèle qu'une fois et réutilisation du modèle à chaque predict
# Class qui initalise le modèle, et qui permet d'appeler la fonction predict.
    
    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            lda_model = pickle.load(f)
        return lda_model
    
    def load_vectorizer(self, vectorizer_path):
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        return vectorizer
    
    def preprocess_data(self, data):
        if isinstance(data, str):
            data_source_list = list()
            data_source_list.append(data)
            data_transformed = pd.DataFrame(data_source_list)
            data_transformed.rename(columns={ data_transformed.columns[0]: "comment" }, inplace = True)
        
        elif isinstance(data, list) or isinstance(data, pd.Series):
            data_transformed = pd.DataFrame(data)
            data_transformed.rename(columns={ data_transformed.columns[0]: "comment" }, inplace = True)

        else:
            raise ValueError(f"Wrong format for data input: {type(data)} is not accepted.\nPass str or list of str.")

        data_transformed['clean_comment'] = data_transformed['comment'].apply(clean_comment)
        data_transformed['tokens'] = data_transformed['clean_comment'].apply(tokenize_comment)
        data_transformed['tokens_without_stopwords'] = data_transformed['tokens'].apply(delete_stopwords)
        data_transformed['tokens_lemm'] = data_transformed['tokens_without_stopwords'].apply(lemmatized_tokens)
        data_transformed['tokens_without_top_words'] = data_transformed['tokens_lemm'].apply(lambda comment: delete_most_frequent_word(comment, self.top_words))

        return data_transformed['tokens_without_top_words']
    

    def transform_data(self, data):
        bow = self.vectorizer.transform(data.apply(" ".join))

        return bow
    
    def predict_topics(self, data):
        topic_probabilities = self.lda_model.transform(data)

        # Trouver le topic le plus probable pour chaque document
        most_probable_topics = topic_probabilities.argmax(axis=1)

        return most_probable_topics, topic_probabilities

