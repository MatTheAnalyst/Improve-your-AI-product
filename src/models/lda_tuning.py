from gensim.models import LdaMulticore, CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation


def tune_lda_coherence(corpus, corpus_for_gensim, dictionary, topic_range, alpha_range, eta_range):
    best_score = 0
    best_model = None
    best_params = {}

    for num_topics in topic_range:
        for alpha in alpha_range:
            for eta in eta_range:
                # Entraînement du modèle LDA avec Gensim
                lda_model = LdaMulticore(corpus=corpus_for_gensim, 
                                         id2word=dictionary, 
                                         num_topics=num_topics, 
                                         random_state=100, 
                                         chunksize=100, 
                                         passes=10, 
                                         alpha=alpha, 
                                         eta=eta)

                # Calcul du score de cohérence
                coherence_model_lda = CoherenceModel(model=lda_model,
                                                     texts=corpus.to_list(), 
                                                     dictionary=dictionary, 
                                                     coherence='c_v')
                
                coherence_score = coherence_model_lda.get_coherence()

                # Mise à jour du meilleur modèle
                if coherence_score > best_score:
                    best_model = lda_model
                    best_params = {'num_topics': num_topics, 'alpha': alpha, 'eta': eta}
                    best_score = coherence_score

    return best_model, best_params, best_score

def tune_lda_perplexity(corpus, topic_range, max_iter=10):
    best_perplexity = float('inf')
    best_model = None
    best_num_topics = 0

    for num_topics in topic_range:
        # Entraînement du modèle LDA
        lda_model = LatentDirichletAllocation(n_components=num_topics, 
                                              max_iter=max_iter, 
                                              learning_method='online', 
                                              random_state=100)
        lda_model.fit(corpus)

        # Calcul de la perplexité
        perplexity = lda_model.perplexity(corpus)
        print(f"N_topics : {num_topics}, perplexity : {perplexity}")
        # Mise à jour du meilleur modèle
        if perplexity < best_perplexity:
            best_model = lda_model
            best_num_topics = num_topics
            best_perplexity = perplexity
            
    return best_model, best_num_topics, best_perplexity
