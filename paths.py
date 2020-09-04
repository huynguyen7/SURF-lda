import os
from pathlib import Path


def reset_file_first_phase(paths):
    for path in paths[1:]:
        os.remove(path=path)
        open(path, 'w').close()


def reset_file_second_phase(paths):
    reset_index = [0, 4, 6, 11]
    for i in range(len(paths)):
        if i not in reset_index:
            os.remove(path=paths[i])
            open(paths[i], 'w').close()


def reset_file_third_phase(paths):
    reset_index = [0, 4, 5, 6, 7, 8, 11, 12]
    for i in range(len(paths)):
        if i not in reset_index:
            os.remove(path=paths[i])
            open(paths[i], 'w').close()


def create_file(path):
    Path(path).touch()
    open(path, 'w').close()
    os.remove(path=path)
    Path(path).touch()
    open(path, 'w').close()


""" Just paths """
def get_paths_first_phase(name):
    comments_path = f'./data/{name}-comments.csv'
    corpus_path = f'./output/corpus/{name}-corpus.pkl'
    dtm_path = f'./output/document-term-matrix/{name}-dtm.pkl'
    stop_words_path = f'./output/stop_words/{name}_stop_cv.pkl'
    words_frequency_first_round = f'./output/eda/first_round/word_freq/{name}.txt'
    words_frequency_second_round = f'./output/eda/second_round/word_freq/{name}.txt'
    top_10_words_first_round = f'./output/eda/first_round/top_10_words/{name}.txt'
    top_10_words_second_round = f'./output/eda/second_round/top_10_words/{name}.txt'
    bigrams_list = f'./output/bigrams/{name}.txt'
    dictionary = f'./output/dictionary/{name}.gensim'
    words_frequency_third_round = f'./output/eda/third_round/word_freq/{name}.txt'
    first_round_cleaned_data = f'./cleaned_data/first_round/{name}/{name}.pkl'

    paths = [comments_path, corpus_path, dtm_path, stop_words_path,
             words_frequency_first_round, words_frequency_second_round,
             top_10_words_first_round, top_10_words_second_round, bigrams_list,
             dictionary, words_frequency_third_round, first_round_cleaned_data]
    reset_file_first_phase(paths)
    return paths


def get_paths_second_phase(name):
    comments_path = f'./data/{name}-comments.csv'
    corpus_path = f'./output/corpus/{name}-corpus.pkl'
    dtm_path = f'./output/document-term-matrix/{name}-dtm.pkl'
    stop_words_path = f'./output/stop_words/{name}_stop_cv.pkl'
    words_frequency_first_round = f'./output/eda/first_round/word_freq/{name}.txt'
    words_frequency_second_round = f'./output/eda/second_round/word_freq/{name}.txt'
    top_10_words_first_round = f'./output/eda/first_round/top_10_words/{name}.txt'
    top_10_words_second_round = f'./output/eda/second_round/top_10_words/{name}.txt'
    bigrams_list = f'./output/bigrams/{name}.txt'
    dictionary = f'./output/dictionary/{name}.gensim'
    words_frequency_third_round = f'./output/eda/third_round/word_freq/{name}.txt'
    first_round_cleaned_data = f'./cleaned_data/first_round/{name}/{name}.pkl'
    second_round_cleaned_data = f'./cleaned_data/second_round/{name}/{name}.pkl'

    # not reset 0, 4, 6, 11
    paths = [comments_path, corpus_path, dtm_path, stop_words_path,
             words_frequency_first_round, words_frequency_second_round,
             top_10_words_first_round, top_10_words_second_round, bigrams_list,
             dictionary, words_frequency_third_round, first_round_cleaned_data,
             second_round_cleaned_data]
    reset_file_second_phase(paths)
    return paths


def get_paths_third_phase(name):
    comments_path = f'./data/{name}-comments.csv'
    corpus_path = f'./output/corpus/{name}-corpus.pkl'
    dtm_path = f'./output/document-term-matrix/{name}-dtm.pkl'
    stop_words_path = f'./output/stop_words/{name}_stop_cv.pkl'
    words_frequency_first_round = f'./output/eda/first_round/word_freq/{name}.txt'
    words_frequency_second_round = f'./output/eda/second_round/word_freq/{name}.txt'
    top_10_words_first_round = f'./output/eda/first_round/top_10_words/{name}.txt'
    top_10_words_second_round = f'./output/eda/second_round/top_10_words/{name}.txt'
    bigrams_list = f'./output/bigrams/{name}.txt'
    dictionary = f'./output/dictionary/{name}.gensim'
    words_frequency_third_round = f'./output/eda/third_round/word_freq/{name}.txt'
    first_round_cleaned_data = f'./cleaned_data/first_round/{name}/{name}.pkl'
    second_round_cleaned_data = f'./cleaned_data/second_round/{name}/{name}.pkl'

    # not reset 0, 4, 5, 6, 7, 8, 11, 12
    paths = [comments_path, corpus_path, dtm_path, stop_words_path,
             words_frequency_first_round, words_frequency_second_round,
             top_10_words_first_round, top_10_words_second_round, bigrams_list,
             dictionary, words_frequency_third_round, first_round_cleaned_data,
             second_round_cleaned_data]
    reset_file_third_phase(paths)
    return paths


def get_paths_without_reset(name):
    comments_path = f'./data/{name}-comments.csv'
    corpus_path = f'./output/corpus/{name}-corpus.pkl'
    dtm_path = f'./output/document-term-matrix/{name}-dtm.pkl'
    stop_words_path = f'./output/stop_words/{name}_stop_cv.pkl'
    words_frequency_first_round = f'./output/eda/first_round/word_freq/{name}.txt'
    words_frequency_second_round = f'./output/eda/second_round/word_freq/{name}.txt'
    top_10_words_first_round = f'./output/eda/first_round/top_10_words/{name}.txt'
    top_10_words_second_round = f'./output/eda/second_round/top_10_words/{name}.txt'
    bigrams_list = f'./output/bigrams/{name}.txt'
    dictionary = f'./output/dictionary/{name}.gensim'
    words_frequency_third_round = f'./output/eda/third_round/word_freq/{name}.txt'
    first_round_cleaned_data = f'./cleaned_datca/first_rounds/{name}/{name}.'
    second_round_cleaned_data = f'./cleaned_data/second_round/{name}/{name}.pkl'
    final_result_folder = f'./output/final_results/{name}/'
    csv_coherence_scores_folder = f'./evaluations/csv_coherence_scores/{name}/'
    lda_visualization_folder = f'./output/lda_visualization/{name}/'
    lda_models_folder = f'./output/lda_models/{name}/'

    paths = [comments_path, corpus_path, dtm_path, stop_words_path,
             words_frequency_first_round, words_frequency_second_round,
             top_10_words_first_round, top_10_words_second_round, bigrams_list,
             dictionary, words_frequency_third_round, first_round_cleaned_data,
             second_round_cleaned_data, final_result_folder, csv_coherence_scores_folder,
             lda_visualization_folder, lda_models_folder]
    return paths
