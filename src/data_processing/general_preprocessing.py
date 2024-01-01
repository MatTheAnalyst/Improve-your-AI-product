import re, string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from pandas import Series


# Supprimer les 10-20 mots les plus fréquents et les valeurs numériques.

def clean_comment(comment):
    """
    Clean and preprocess a comment string.
    
    Args:
        comment: The comment string to be cleaned.
    Returns:
        A cleaned and preprocessed version of the comment.
    """
    if not comment or not isinstance(comment, str):
        print(comment)
        return ""
    
    # Convertir en minuscules
    comment = comment.lower()

    # HTML
    comment = re.sub(r'<.*?>', '', comment)

    # ASCII
    comment = re.sub(r'[^\x00-\x7F]+', '', comment)

    # Numbers
    comment = re.sub(r'[0-9]*', '', comment)

    # Removes any words with 3 or more repeated letters
    comment = re.sub(r"(.)\\1{2,}", '', comment)

    # Removes any remaining single letter words
    comment = re.sub(r"\\b(.)\\b", '', comment)

    # Supprimer la ponctuation
    comment = comment.translate(str.maketrans('', '', string.punctuation))

    return comment
    
def tokenize_comment(comment):
    return word_tokenize(comment)

def delete_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def lemmatized_tokens(tokens):
    pos_tags = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos=get_wordnet_pos(pos)) for word, pos in pos_tags]

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def most_frequent_word(corpus, n_top_words):
    joined_corpus = " ".join([" ".join(word) for word in corpus.values])
    most_frequent_word = list(Series(joined_corpus.split()).value_counts().head(n_top_words).index)
    return most_frequent_word

def delete_most_frequent_word(comment, top_words:list):
    return [word for word in comment if word not in top_words]

def unique_word(corpus):
    joined_corpus = " ".join([" ".join(word) for word in corpus.values])
    joined_corpus = Series(joined_corpus.split()).value_counts()
    one_word = list(joined_corpus.loc[(joined_corpus == 1)].index)
    return one_word

def delete_unique_word(comment, one_words:list):
    return [word for word in comment if word not in one_words]