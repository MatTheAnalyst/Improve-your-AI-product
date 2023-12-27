import re, string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

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

    # Number
    comment = re.sub(r'[0-9]*', '', comment)

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