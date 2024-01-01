from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from pathlib import Path
from numpy import abs, array, argmin, min, max

def tune_lda(corpus, topic_range, max_iter=10):
    best_perplexity = float('inf')
    best_log_likelihood = float('-inf')
    best_model = None
    best_num_topics = 0

    perplexities = []
    log_likelihoods = []

    for num_topics in topic_range:

        lda_model = train_lda(corpus, num_topics, max_iter)
        perplexity = lda_model.perplexity(corpus)
        log_likelihood = lda_model.score(corpus)
        perplexities.append(perplexity)
        log_likelihoods.append(log_likelihood)

    perplexities_normalized = array([(value - min(perplexities)) / (max(perplexities) - min(perplexities)) for value in perplexities])
    log_likelihoods_normalized = array([(value - min(log_likelihoods)) / (max(log_likelihoods) - min(log_likelihoods)) for value in log_likelihoods])
    
    # Calculer la différence absolue entre les deux métriques
    differences = abs(array(perplexities_normalized) - array(log_likelihoods_normalized))

    # Trouver l'index de la différence minimale
    min_difference_index = argmin(differences)

    best_num_topics = topic_range[min_difference_index]
    best_model = train_lda(corpus, best_num_topics, max_iter)
    best_perplexity = best_model.perplexity(corpus)
    best_log_likelihood = best_model.score(corpus)

    # Tracer les courbes
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Nombre de topics')
    ax1.set_ylabel('Perplexité', color='tab:blue')
    ax1.plot(topic_range, perplexities, label='Perplexité', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.axvline(x=best_num_topics, color='r', linestyle='--', label='Meilleur nombre de topics')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Log Likelihood', color='tab:red')  # we already handled the x-label with ax1
    ax2.plot(topic_range, log_likelihoods, label='Log Likelihood', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Évolution de la Perplexité et du Log Likelihood')
    destination = (Path(__file__).parents[1] / 'visualization/lda_optimization.png').absolute()
    plt.savefig(str(destination))
    
    return best_model, best_num_topics, best_perplexity, best_log_likelihood

# best_model, best_num_topics, best_perplexity, best_log_likelihood = tune_lda(corpus, range(2, 15), max_iter=10)

def train_lda(corpus, n_components, max_iter=10):
    lda_model = LatentDirichletAllocation(n_components=n_components, max_iter=max_iter, learning_method='online', random_state=100)
    lda_model.fit(corpus)
    return lda_model