def gensim_print_top_words_topics(model, n_topics, n_top_words):
    for i in range(n_topics):
        words = [word for word, prob in model.show_topic(i, n_top_words)]
        print(f"Topic {i+1}: {' '.join(words)}")


def sklearn_print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = f"Topic #{topic_idx}: "
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)