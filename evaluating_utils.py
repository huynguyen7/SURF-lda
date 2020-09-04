import os
import pickle
import re
import numpy as np
import pandas
from gensim import models
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from tqdm import tqdm
from gensim.models.callbacks import PerplexityMetric, ConvergenceMetric, CoherenceMetric

from main_utils import get_LDA_model, save_lda_model, get_LDA_model_multi_cores
from paths import create_file
from visualizing_utils import print_topic_coherence_to_text_file, print_perplexity_to_text_file, \
    print_topics_to_text_file, visualize_LDA, graph_coherence_scores_num_topics
import logging


def get_models_log_perplexity_covergence_coherence(paths, num_topics, iterations, passes, log_path):
    create_file(log_path)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        filename=log_path, filemode='a',
                        level=logging.NOTSET)

    with open(paths[1], 'rb') as f:
        corpus = pickle.load(f)  # bag-of-words

    dictionary = Dictionary.load(paths[9])
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    perplexity_logger = PerplexityMetric(corpus=corpus, logger='shell')
    convergence_logger = ConvergenceMetric(logger='shell')
    coherence_cv_logger = CoherenceMetric(corpus=corpus, logger='shell', coherence='u_mass')

    for iteration in tqdm(iterations):
        logging.debug(f'Start of model: {iteration} iterations')

        # Create model with callbacks argument uses list of created callback loggers
        model = models.ldamodel.LdaModel(corpus=corpus,
                                         id2word=dictionary,
                                         num_topics=num_topics,
                                         eval_every=1,
                                         chunksize=5932,
                                         passes=passes,
                                         random_state=100,
                                         iterations=iteration,
                                         callbacks=[convergence_logger, perplexity_logger, coherence_cv_logger])

        logging.debug(f'End of model: {iteration} iterations')
        if not os.path.exists(f"./evaluations/graph_scores/{num_topics}_topics_{iterations[-1]}_iterations_{passes}_passes/models/lda_{iteration}i{passes}p_models/"):
            os.makedirs(f"./evaluations/graph_scores/{num_topics}_topics_{iterations[-1]}_iterations_{passes}_passes/models/lda_{iteration}i{passes}p_models/")
        model.save(f"./evaluations/graph_scores/{num_topics}_topics_{iterations[-1]}_iterations_{passes}_passes/models/lda_{iteration}i{passes}p_models/lda_{iteration}i{passes}p.gensim")


# Function to detect relevant numbers in log
def find_doc_convergence(num_topics, iteration, passes, log):
    # Regex to bookend log for iteration - choose last occurrence
    end_slice = re.compile(f"End of model:.*? {iteration} iterations")
    end_matches = [end_slice.findall(l) for l in open(log)]  # Find all end matches string pattern
    iteration_end = [i for i, x in enumerate(end_matches) if x]
    iteration_end = iteration_end[-1]
    start_slice = re.compile(f"Start of model:.*? {iteration} iterations")
    start_matches = [start_slice.findall(l) for l in open(log)]  # Find all start matches string pattern
    start_options = [i for i, x in enumerate(start_matches) if x]
    start_options = [item for item in start_options if item < iteration_end]
    iteration_start = max(start_options)
    iteration_bookends = [iteration_start, iteration_end]

    # Regex to find documents converged figures
    p = re.compile(": (\d+)\/\d")
    matches = [p.findall(l) for l in open(log)]
    matches = matches[iteration_bookends[0]:iteration_bookends[1]]
    matches = [m for m in matches if len(m) > 0]

    # Unlist internal lists and turn into numbers
    matches = [m for sublist in matches for m in sublist]
    matches = [float(m) for m in matches]
    return matches


def get_alpha_beta_lists():
    # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.3))
    alpha.append('symmetric')
    alpha.append('asymmetric')
    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    beta.append('symmetric')
    return alpha, beta


def evaluate_LDA_models(paths, scores_path, corpus, id2word, fr, to, step, passes):
    coherence_values = []
    models_list = []
    output_path = paths[13]
    graph_path = scores_path + str(fr) + '-' + str(to) + '-' + str(passes) + '.pdf'

    for num_topics in range(fr, to, step):
        print('\nStart to EDA with: %d topics and %d passes' % (num_topics, passes))
        model = get_LDA_model(paths, num_topics, passes)
        with open(paths[1], 'rb') as f:
            corpus = pickle.load(f)

        text_file_path = output_path + str(num_topics) + '-' + str(passes) + '.txt'
        # topic coherence and perplexity
        print_topic_coherence_to_text_file(text_file_path, num_topics, model,
                                           corpus, paths[9])
        print_perplexity_to_text_file(text_file_path, model, corpus)

        # lda outputs all topics to text file
        num_words = 50
        print_topics_to_text_file(model, text_file_path, num_words)

        # save lda model
        save_lda_model(paths, model, num_topics, passes)

        # visualize to html file
        visualize_LDA(paths, num_topics, passes)
        models_list.append(model)

        coherence_model_lda = CoherenceModel(model=model, corpus=corpus, dictionary=id2word, coherence='u_mass')
        coherence_values.append(coherence_model_lda.get_coherence())
    graph_coherence_scores_num_topics(coherence_values, graph_path, fr, to, step)

    return models_list, coherence_values


def evaluate_LDA_models_multicores(paths, scores_path, corpus, id2word, fr, to, step, passes):
    alpha, beta = get_alpha_beta_lists()
    models_list = []
    corpus_sets = [# ClippedCp(corpus, num_of_docs * 0.25), ClippedCp(corpus, num_of_docs * 0.5), ClippedCp(corpus, num_of_docs * 0.75),
        corpus]
    corpus_title = [#'25% Corpus', '50% Corpus', '75% Corpus',
                    '100% Corpus']  # This match with the corpus-sets
    grid = {'Validation_Set': {}}
    model_results = {
        'Validation_Set': [],
        'Topics': [],
        'Alpha': [],
        'Beta': [],
        'Coherence_Score': []
    }
    topics_range = range(fr, to, step)

    if 1 == 1:
        pbar = tqdm.tqdm(total=(len(beta)*len(alpha)*len(topics_range)*len(corpus_title)))  # Progress bar to keep track
        for i in range(len(corpus_sets)):  # iterate through validation corpuses
            for a in alpha:  # iterate through alpha values
                for b in beta:  # iterate through beta values
                    # coherence_values = []  # Empty coherence_values every time for graphing
                    for num_topics in topics_range:  # iterate through number of topics
                        lda_model = get_LDA_model_multi_cores(paths, corpus, id2word, num_topics=num_topics, passes=passes, a=a, b=b)
                        models_list.append(lda_model)

                        text_file_path = paths[13] + str(num_topics) + '-' + str(passes) + '.txt'

                        # visualize to html file
                        visualize_LDA(paths, num_topics, passes, alpha=a, beta=b)
                        print_topic_coherence_to_text_file(text_file_path, num_topics, lda_model,
                                                           corpus, paths[9])
                        print_perplexity_to_text_file(text_file_path, lda_model, corpus)
                        coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, dictionary=id2word,
                                                             coherence='u_mass')
                        # coherence_values.append(coherence_model_lda.get_coherence())

                        # Save the model results with alpha, beta and coherence score
                        model_results['Validation_Set'].append(corpus_title[i])
                        model_results['Topics'].append(num_topics)
                        model_results['Alpha'].append(a)
                        model_results['Beta'].append(b)
                        model_results['Coherence_Score'].append(coherence_model_lda.get_coherence())
                        pbar.update(1)

                    # Save coherence scores graph, this is crashed #FIXME
                    # graph_path = scores_path + str(a) + '_alpha_' + str(b) + '_beta_' + str(fr) + '-' + str(to-1) + '-' + str(passes) + '.pdf'
                    # graph_coherence_scores_num_topics(coherence_values, graph_path, fr, to, step)
        csv_results_path = paths[14] + str(fr) + '-' + str(to-1) + '-' + str(passes) + '.csv'
        create_file(csv_results_path)
        rs = pandas.DataFrame(model_results)
        rs.sort_values(by=['Coherence_Score'], inplace=True, ascending=False)
        rs.to_csv(csv_results_path, index=False)
        pbar.close()
    return models_list