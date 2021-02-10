import warnings
import sys
import pickle
import pandas
from gensim import models
from tqdm import tqdm
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from cleaning_utils import first_round, get_stop_words, second_round
from main_utils import get_corpus_data_frame, add_bigrams_to_docs, save_as_pickle_for_lda, \
    get_LDA_model, save_lda_model, get_combined_data, get_LDA_mallet_model, alarm, get_iteration_list, \
    save_lda_mallet_model
from paths import *
from sklearn.feature_extraction.text import CountVectorizer
from visualizing_utils import show_bigrams_freq, get_words_bigrams_frequency, get_words_frequency, \
    print_topic_coherence_to_text_file, print_perplexity_to_text_file, print_topics_to_text_file, visualize_LDA, \
    visualize_LDA_mallet, graphs_convergence_perplexity_coherence
from evaluating_utils import evaluate_LDA_models_multicores, get_models_log_perplexity_covergence_coherence, \
    find_doc_convergence


def clean_data_first_phase(paths):
    df = pandas.read_csv(paths[0], encoding="ISO-8859-1", header=0)

    pandas.set_option('max_colwidth', 1000)
    submission_texts_df = df.Submission_Text
    submission_numbers_df = df.Submission_Number

    cv = CountVectorizer(stop_words=stopwords.words('english'))  # Remove some basic stop words
    combined_data = get_combined_data(submission_numbers_df, submission_texts_df)
    data_df = get_corpus_data_frame(combined_data)  # Remove dups docs

    first_round_cleaned_data = first_round(paths, cv, data_df, stopwords.words('english'))
    print("Start to EDA first round")
    EDA(paths, paths[4], paths[6], first_round_cleaned_data)
    print("Finished EDA first round")


def clean_data_second_phase(paths):
    first_round_cleaned_data = pandas.read_pickle(paths[11])
    stop_words = get_stop_words()
    cv = CountVectorizer(stop_words=stop_words)  # Removing words from stop words list
    second_round_cleaned_data = second_round(paths, first_round_cleaned_data, cv, stop_words)
    print("Start to EDA second round")
    EDA(paths, paths[5], paths[7], second_round_cleaned_data)
    print("Finished EDA second round")

    print("Getting bigrams")
    show_bigrams_freq(second_round_cleaned_data, paths[8])
    print("Finished getting bigrams")


def clean_data_third_phase(paths, bigram_threshold, no_below, no_above):
    # adding bigrams to docs
    # Note that there is another cleaning round inside
    # Add bigrams to docs (only ones that appear <bigram_threshold> times or more).
    second_round_cleaned_data = pandas.read_pickle(paths[12])
    docs = add_bigrams_to_docs(second_round_cleaned_data, min_count=bigram_threshold)

    # Get words-bigrams freq
    get_words_bigrams_frequency(docs, paths[10])

    # Remove words and bigrams that appear in less than <no_below> documents
    # Remove words and bigrams that appear more than (<no_above> x 100)% documents
    # then save as pickle files.
    save_as_pickle_for_lda(docs, paths, no_below=no_below, no_above=no_above)


""" Exploratory Data Analyzing (EDA) """
def EDA(paths, word_freq_path, top_10_words_path, data_cleaned):
    dtm = pandas.read_pickle(paths[2])
    tdm = dtm.transpose()  # term-document matrix

    "get the frequency of each word."
    fdist = get_words_frequency(data_cleaned, word_freq_path)
    # get_words_frequency2(dtm, word_freq_path)

    "get 10 words which appear most in each comment."
    # get_top_10_words_each_comment(tdm, top_10_words_path)

    "plot top 10 words which appear most in all comments"
    # fdist[:9].plot()


def LDA(paths, output_path, num_topics, iterations, chunksize, passes, minimum_probability):
    print('\nStart to LDA with: %d topics and %d iterations' % (num_topics, iterations))
    lda_model = get_LDA_model(paths, num_topics, iterations, chunksize, passes, minimum_probability)
    with open(paths[1], 'rb') as f:
        corpus = pickle.load(f)

    text_file_path = output_path + str(num_topics) + '-' + str(iterations) + '.txt'
    # topic coherence and perplexity
    print_topic_coherence_to_text_file(text_file_path, num_topics, lda_model,
                                       corpus, paths[9])
    print_perplexity_to_text_file(text_file_path, lda_model, corpus)

    # lda outputs all topics to text file
    num_words = 50
    print_topics_to_text_file(lda_model, text_file_path, num_words)

    # save lda model
    save_lda_model(paths, lda_model, num_topics, iterations)

    # visualize to html file
    visualize_LDA(paths, num_topics, iterations)

    return lda_model


def LDA_with_mallet(paths, output_path, num_topics, iterations, minimum_probability):
    print('\nStart to LDA with: %d topics and %d iterations' % (num_topics, iterations))
    lda_model = get_LDA_mallet_model(paths, num_topics=num_topics, iterations=iterations, minimum_probability=minimum_probability)
    with open(paths[1], 'rb') as f:
        corpus = pickle.load(f)

    text_file_path = output_path + str(num_topics) + '-' + str(iterations) + '.txt'
    # topic coherence and perplexity
    print_topic_coherence_to_text_file(text_file_path, num_topics, lda_model,
                                       corpus, paths[9])

    # lda outputs all topics to text file
    num_words = 50  # write <num_words> words/topic to text file
    print_topics_to_text_file(lda_model, text_file_path, num_words)

    # save lda model
    save_lda_mallet_model(paths, lda_model, num_topics, iterations)

    # visualize to html file
    visualize_LDA_mallet(paths, num_topics, iterations)

    return lda_model


def evaluate_with_coherence_score(paths, scores_path, fr, to, step, passes):
    to = to + 1
    with open(paths[1], 'rb') as f:
        corpus = pickle.load(f)
    dictionary = Dictionary.load(paths[9])
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    # Using original LDA gensim
    # evaluate_LDA_models(paths scores_path, corpus, id2word, fr, to, step, passes)

    # Using LDA gensim multicore
    evaluate_LDA_models_multicores(paths, scores_path, corpus, id2word, fr, to, step, passes)


def evaluate_with_callbacks(paths, num_topics, iterations, passes):
    sys.setrecursionlimit(100000)

    if not os.path.exists(f'./evaluations/graph_scores/{num_topics}_topics_{iterations[-1]}_iterations_{passes}_passes/'):
        os.makedirs(f'./evaluations/graph_scores/{num_topics}_topics_{iterations[-1]}_iterations_{passes}_passes/')
    if not os.path.exists(f'./evaluations/graph_scores/{num_topics}_topics_{iterations[-1]}_iterations_{passes}_passes/models/'):
        os.makedirs(f'./evaluations/graph_scores/{num_topics}_topics_{iterations[-1]}_iterations_{passes}_passes/models/')
    if not os.path.exists(f"./evaluations/graph_scores/{num_topics}_topics_{iterations[-1]}_iterations_{passes}_passes/graphs/"):
        os.makedirs(f"./evaluations/graph_scores/{num_topics}_topics_{iterations[-1]}_iterations_{passes}_passes/graphs/")
    log_path = f'./evaluations/graph_scores/{num_topics}_topics_{iterations[-1]}_iterations_{passes}_passes/tmp_log.txt'

    get_models_log_perplexity_covergence_coherence(paths, num_topics, iterations, passes, log_path=log_path)
    all_metrics = pandas.DataFrame()

    for iteration in tqdm(iterations):
        model = models.ldamodel.LdaModel.load(f"./evaluations/graph_scores/{num_topics}_topics_{iterations[-1]}_iterations_{passes}_passes/models/lda_{iteration}i{passes}p_models/lda_{iteration}i{passes}p.gensim")
        df = pandas.DataFrame.from_dict(model.metrics)
        df['docs_converged'] = find_doc_convergence(num_topics=num_topics, iteration=iteration, passes=passes, log=log_path)
        df['iterations'] = iteration
        df['topics'] = num_topics
        df = df.reset_index().rename(columns={'index': 'pass_num'})
        all_metrics = pandas.concat([all_metrics, df])

    graphs_convergence_perplexity_coherence(all_metrics=all_metrics, num_topics=num_topics, iterations=iterations, passes=passes)
    alarm(repeat=5)


warnings.filterwarnings("ignore", category=DeprecationWarning)


""" Clean data and save as pkl files """
# clean_data_first_phase(get_paths_first_phase('all'))
# clean_data_second_phase(get_paths_second_phase('all'))
clean_data_third_phase(get_paths_third_phase('all'), bigram_threshold=25, no_below=20, no_above=0.5)


""" Evaluate LDA models"""
# evaluate_with_coherence_score(get_paths_without_reset('all'),
#                               './evaluations/coherence_scores_graphs/all/',
#                               fr=5, to=50, step=1, passes=100)

# evaluate_with_coherence_score(get_paths_without_reset('test'),
#                               './evaluations/coherence_scores_graphs/test/',
#                               fr=5, to=8, step=1, passes=2)


"""Evaluate LDA models with perplexity scores, coherence values, convergence"""
# num_topics = 10
# passes = 50
#
# evaluate_with_callbacks(get_paths_without_reset('all'),
#                         iterations=get_iteration_list(start_iteration=10, max_iteration=100, step=10),
#                         passes=passes, num_topics=num_topics)

# evaluate_with_callbacks(get_paths_without_reset('test'),
#                         iterations=get_iteration_list(start_iteration=10, max_iteration=100, step=10),
#                         passes=passes, num_topics=num_topics)


""" Topic modeling with LDA """
# iterations = 200
# LDA(get_paths_without_reset('all'), './output/final_results/all/', num_topics=10, iterations=iterations, chunksize=1592, passes=50, minimum_probability=0.2)
# LDA(get_paths_without_reset('all'), './output/final_results/all/', num_topics=15, iterations=iterations, chunksize=1592, passes=50, minimum_probability=0.2)
# LDA(get_paths_without_reset('all'), './output/final_results/all/', num_topics=20, iterations=iterations, chunksize=1592, passes=50, minimum_probability=0.2)
# LDA(get_paths_without_reset('all'), './output/final_results/all/', num_topics=25, iterations=iterations, chunksize=1592, passes=50, minimum_probability=0.2)
# LDA(get_paths_without_reset('all'), './output/final_results/all/', num_topics=30, iterations=iterations, chunksize=1592, passes=50, minimum_probability=0.2)
# alarm(repeat=5)

# LDA(get_paths_without_reset('test'), './output/final_results/get_LDA_model_multi_corestest/', 5, 20)

minimum_probability = 0
iterations = 200
# LDA_with_mallet(get_paths_without_reset('all'), './output/final_results/all/', num_topics=20, iterations=iterations, minimum_probability=minimum_probability)
# LDA_with_mallet(get_paths_without_reset('all'), './output/final_results/all/', num_topics=25, iterations=iterations, minimum_probability=minimum_probability)
# LDA_with_mallet(get_paths_without_reset('all'), './output/final_results/all/', num_topics=30, iterations=iterations, minimum_probability=minimum_probability)
#LDA_with_mallet(get_paths_without_reset('all'), './output/final_results/all/', num_topics=35, iterations=iterations, minimum_probability=minimum_probability)
alarm(repeat=5)
