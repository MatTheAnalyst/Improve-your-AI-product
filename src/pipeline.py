import pickle
from pathlib import Path
import pandas as pd
from data_processing.general_preprocessing import clean_comment, tokenize_comment, delete_stopwords, lemmatized_tokens, most_frequent_word, delete_most_frequent_word, unique_word, delete_unique_word

file_path = Path(__file__)
top_words_path = str(file_path.parent / "models/top_words.pkl")
vectorizer_path = str(file_path.parent / "models/vectorizer.pkl")
model_path = str(file_path.parent / "models/lda_model.pkl")

with open(top_words_path, 'rb') as f:
    top_words = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

with open(model_path, 'rb') as f:
    lda_model = pickle.load(f)

data_source = ["I went to see dr swing since I was 12 week pregnancy. The first thing I noticed is that seems the dr is very busy and she does not have enough time to carefully listen to your situation and concerns. Every routine check i need to wait outside for more than 30 min and about another 20 min in the exam room. \n\nThe thing started to happen since I am 27 week pregnancy and this is the first time the dr told me that my baby was smaller than the average for a week. I asked her if I need to worry, she said no. But she changed the exam to every two weeks instead of one month, and Everytime after that I had ton do an ultrasound(and this made my bill is very high.) and finally when I was 32 week, she said she referred me to another dr to see the baby why she was so small. I doubled checked with her to see if my baby was small because I am Asia but she said no. So I went to see the dr, turned out that doc is the one who treat high risk pregnant women, but dr swing was never told me that my situation was that bad. Then the new doctor did another ultrasound and told me that the baby's size was not a problem. The new problem is that she had double bubble, and there was 90% that she was don. I did five ultrasounds at dr swing, and none of them told me that my baby had this serious issue and needs a surgery immediately after she was born.  All of those things made me blood pressure increased too much and i had no choice to do c section to get y baby out when she was only 35 weeks. But, after she was born and did x Ray and turned out my baby was perfectly healthy!!! So all of those things were misdiagnosed."]

data_transformed = pd.DataFrame(data_source, columns=["source"])

data_transformed['clean_comment'] = data_transformed['source'].apply(clean_comment)
data_transformed['tokens'] = data_transformed['clean_comment'].apply(tokenize_comment)
data_transformed['tokens_without_stopwords'] = data_transformed['tokens'].apply(delete_stopwords)
data_transformed['tokens_lemm'] = data_transformed['tokens_without_stopwords'].apply(lemmatized_tokens)
data_transformed['tokens_without_top_words'] = data_transformed['tokens_lemm'].apply(lambda comment: delete_most_frequent_word(comment, top_words))

bow = vectorizer.transform(data_transformed['tokens_without_top_words'].apply(" ".join))
feature_names = vectorizer.get_feature_names_out()

# Transformation avec LDA
topic_probabilities = lda_model.transform(bow)

# Trouver le topic le plus probable pour chaque document
most_probable_topics = topic_probabilities.argmax(axis=1)

# Ajouter les résultats au DataFrame
data_transformed['most_probable_topic'] = most_probable_topics
data_transformed['topic_probability'] = topic_probabilities.max(axis=1)

# Afficher les résultats
print(data_transformed[['source', 'most_probable_topic', 'topic_probability']])

# Obtenir les mots clés pour chaque topic
n_top_words = 10
for topic_idx, topic in enumerate(lda_model.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))


