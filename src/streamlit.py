import streamlit as st
from pandas import DataFrame
from numpy import round
from pipeline import PipelineLDA

st.title("Détection des sujets d'insatisfaction")

input_text = str(st.text_input('Entrer le commentaire pour afficher son topic :'))

if input_text:
    config = {
        'model_path': "models/lda_model.pkl",
        'vectorizer_path': "models/vectorizer.pkl",
        'top_words': "models/top_words.json",
        'interpreted_topics': "models/dic_topics.json"
    }
    try:
        pipeline = PipelineLDA(config)
        print(pipeline.interpreted_topics)
    except ValueError as e:
        print(f"Error : {e}")
    
"""
    try:
        results = lda(input_text)
        print(results['topic_interpretated'])
        print("-"*50)
        print(results)
    except ValueError as e:
        print(f"Error : {e}")

# La probabilité d'appartenance aux 3 premiers topics


        preprocessed_data = pipeline.preprocess_data(input_text)
        transformed_data = pipeline.transform_data(preprocessed_data)
        most_probable_topics, topic_probabilities = pipeline.predict_topics(transformed_data)
        results = DataFrame(data={"Topics": pipeline.interpreted_topics.values(),"Probabilites":round(topic_probabilities[0],2)})
        print(results)
        print("-"*50)
        print("Le topic le plus probable est :\n{pipeline.interpreted_topics[most_probable_topics.item()]}")

"""