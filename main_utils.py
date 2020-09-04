import os
import pandas
import pickle
import scipy.sparse
from gensim import matutils, models
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases
from nltk import word_tokenize
from paths import create_file
from csv import writer
from gensim.models.wrappers import LdaMallet


def append_row_to_csv(file, row):  # row is a string
    with open(file, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(row)


def append_row_to_text_file(string, path):
    with open(path, "a") as file_object:
        file_object.write("\n" + string)


def transform_to_list(data):
    docs = data.Submission_Text.tolist()
    for doc_id in range(len(docs)):
        docs[doc_id] = word_tokenize(docs[doc_id])  # Split into words.
    return docs


def get_retained_words_list():
    # Note that this list used for add_bigrams_to_docs()
    # remove words in the list,
    # but keep the bigrams which is created by the list
    words_list = [
        'millenium', 'millennium', 'millenniums', 'milleniums',
        'bulk', 'bulks', 'washington', 'state', 'states',
        'terminal', 'terminals', 'cowlitz', 'county', 'counties',
        'port', 'ports', 'longview', 'longviews',
        'project', 'projects', 'coal', 'coals'
    ]
    return words_list


def save_lda_model(paths, lda_model, num_topics, passes, alpha=None, beta=None):
    # save lda model
    if alpha is not None and beta is not None:
        lda_model_path = paths[16] + str(alpha) + '_alpha_' + str(beta) + '_beta_' + str(num_topics) + '-' + str(
            passes) + '.gensim'
    else:
        lda_model_path = paths[16] + str(num_topics) + '-' + str(passes) + '.gensim'
    create_file(lda_model_path)
    lda_model.save(lda_model_path)


def add_bigrams_to_docs(data, min_count):
    docs = transform_to_list(data)
    removed_word_list = get_retained_words_list()
    bigram = Phrases(docs, min_count=min_count)

    final_docs = []
    for doc_id in range(len(docs)):
        i = 0
        final_docs.append([])
        while i < (len(docs[doc_id]) - 2):
            for token in bigram[docs[doc_id][i], docs[doc_id][i + 1]]:
                if token not in removed_word_list:
                    final_docs[doc_id].append(token)  # remove word but not remove its bi_grams
            i += 1
    return final_docs


def save_as_pickle_for_lda(docs, paths, no_below, no_above):
    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)

    # Filter out words that occur less than [no_below] documents
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    dictionary.save(paths[9])

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    pickle.dump(corpus, open(paths[1], "wb"))  # save as corpus


def get_LDA_mallet_model(paths, num_topics, passes):
    with open(paths[1], 'rb') as f:
        corpus = pickle.load(f)  # sparse terms (sparse matrix form of corpus)

    dictionary = Dictionary.load(paths[9])
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    path_to_mallet = '~/mallet-2.0.8/'
    os.environ.update({'MALLET_HOME': path_to_mallet})
    path_to_mallet_bin = '~/mallet-2.0.8/bin/mallet'

    model = LdaMallet(
        mallet_path=path_to_mallet_bin, corpus=corpus, num_topics=num_topics,
        id2word=id2word, workers=2, iterations=passes, topic_threshold=0
    )

    return model


def get_LDA_model(paths, num_topics, iterations, chunksize, passes, minimum_probability):
    with open(paths[1], 'rb') as f:
        corpus = pickle.load(f)  # bag-of-words

    dictionary = Dictionary.load(paths[9])
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = models.LdaModel(
        corpus=corpus, id2word=id2word, alpha='auto',
        eta='auto', iterations=iterations, num_topics=num_topics,
        per_word_topics=True, minimum_probability=minimum_probability,
        chunksize=chunksize, eval_every=None, passes=passes,
    )

    return model

def get_combined_data(submission_numbers_df, submission_texts_df):
    combined_data = {}  # empty dictionary
    i = 0
    for sub_num in submission_numbers_df.values:
        if type(submission_texts_df[i]) is float:  # Remove empty comment
            i = i + 1
        else:
            combined_data[sub_num] = [submission_texts_df[i]]
            i = i + 1
    return combined_data


def get_corpus_data_frame(combined_data):
    data_df = pandas.DataFrame.from_dict(combined_data).transpose()
    data_df.columns = ['Submission_Text']
    data_df.drop_duplicates(subset='Submission_Text', keep='first', inplace=True)  # Remove duplicated comments
    data_df = data_df.sort_index()
    return data_df


def get_LDA_model_multi_cores(paths, corpus, id2word, num_topics, passes, a=None, b=None):
    if a is None and b is None:
        lda_model = models.LdaMulticore(
            corpus=corpus, id2word=id2word,
            passes=passes, num_topics=num_topics, workers=4,
            chunksize=100, per_word_topics=True, minimum_probability=0.0
        )
    else:
        lda_model = models.LdaMulticore(
            corpus=corpus, id2word=id2word,
            passes=passes, num_topics=num_topics, workers=4,
            alpha=a, eta=b, chunksize=100, per_word_topics=True, minimum_probability=0.0
        )
    save_lda_model(paths, lda_model, num_topics=num_topics, passes=passes, alpha=a, beta=b)
    return lda_model


def get_iteration_list(start_iteration, max_iteration, step):
    iterations = list()
    while start_iteration <= max_iteration:
        iterations.append(start_iteration)
        start_iteration = start_iteration + step
    return iterations

def alarm(repeat=30):
    i = 0
    while i < repeat:
        os.system('say "your program has finished"')
        i = i + 1
