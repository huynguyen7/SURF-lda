import os

import pyLDAvis
import pandas as pd
import tqdm
import math
import pandas
import pickle
import warnings
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
from gensim.models.wrappers import LdaMallet
from visualizing_utils import dominant_topics, topics_proportion
from main_utils import alarm, append_row_to_text_file, get_combined_data
from gensim import models
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from pandas import read_csv
from pyLDAvis import gensim
from gensim.models import LdaModel
from paths import create_file, get_paths_without_reset


warnings.filterwarnings("ignore", category=DeprecationWarning)

""" Only to to load models path """
corpus_path = './output/corpus/all-corpus.pkl'
dataset_csv_path = './turn-in/dataset/dataset.csv'
bigram_threshold = 25  # CHANGE THIS VALUE
iteration = 200  # CHANGE THIS VALUE
num_topics = [30]  # CHANGE THIS VALUE
models_path = list()
for num_topic in num_topics:
    models_path.append(f'./turn-in/{bigram_threshold}/lda_models/{num_topic}-{iteration}.gensim')


""" Get coherence score for each model """
def coherence_scores_to_csv(models_path, num_topics):
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)

    index = 0
    df = pd.DataFrame()
    output_path = f'./turn-in/{bigram_threshold}/topic_coherence.csv'
    create_file(output_path)

    while index < len(models_path):
        lda_model = LdaModel.load(models_path[index])
        if type(lda_model) is LdaMallet:
            lda_model = models.wrappers.ldamallet.malletmodel2ldamodel(lda_model, iterations=iteration)
        top_topics = lda_model.top_topics(corpus)
        avg_topic_coherence = sum([topic[1] for topic in top_topics]) \
                              / num_topics[index]
        df = df.append(pd.Series(
            [num_topics[index], avg_topic_coherence]), ignore_index=True)
        index = index + 1
    df.columns = ['Num_Topics', 'Coherence_Score']
    df.to_csv(output_path, index=False)


""" Calculate entropy for lda model """
def calculate_entropy(bigram_threshold):  # output to csv files.
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)

    index = 0
    dataset = pandas.read_csv(dataset_csv_path)
    for model_path in models_path:
        lda_model = LdaModel.load(model_path)
        df = pd.DataFrame()
        pbar = tqdm.tqdm(total=len(lda_model[corpus]))

        for i, row in enumerate(lda_model[corpus]):
            topic_dist = sorted(row[0], key=lambda x: (x[1]), reverse=True)
            rs_string = ''
            topic_entropy = 0
            for topic in topic_dist:
                rs_string = rs_string + 'Topic ' + str(topic[0] + 1) + ': ' + str(topic[1]) + '; '
                topic_entropy = topic_entropy + (-math.log2(topic[1]))
            df = df.append(pd.Series(
                [str(i), dataset['Submission_Num'][i], rs_string, str(topic_entropy), dataset['Submission_Text'][i]]),
                ignore_index=True)
            pbar.update(1)
        df.columns = ['Document_No', 'Submission_Num', 'Probabilities', 'Entropy',
                      'Submission_Text']

        csv_file_result_path = f'./turn-in/{bigram_threshold}/model_entropy/{num_topics[index]}.csv'
        index = index + 1
        create_file(csv_file_result_path)
        df.to_csv(csv_file_result_path, index=False)
        pbar.close()


def calculate_entropy_mallet_models():  # output to csv files.
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)

    index = 0
    dataset = pandas.read_csv(dataset_csv_path)
    for model_path in models_path:
        lda_model = LdaMallet.load(model_path)
        lda_model = models.wrappers.ldamallet.malletmodel2ldamodel(lda_model, iterations=iteration)

        df = pd.DataFrame()
        pbar = tqdm.tqdm(total=len(lda_model[corpus]))

        for i, row in enumerate(lda_model[corpus]):
            topic_dist = sorted(row, key=lambda x: (x[1]), reverse=True)
            rs_string = ''
            topic_entropy = 0
            for topic in topic_dist:
                rs_string = rs_string + 'Topic ' + str(topic[0] + 1) + ': ' + str(topic[1]) + '; '
                topic_entropy = topic_entropy + (-math.log2(topic[1]))
            df = df.append(pd.Series(
                [str(i), dataset['Submission_Num'][i], rs_string, str(topic_entropy), dataset['Submission_Text'][i]]),
                ignore_index=True)
            pbar.update(1)
        df.columns = ['Document_No', 'Submission_Num', 'Probabilities', 'Entropy',
                      'Submission_Text']

        csv_file_result_path = f'./turn-in/{bigram_threshold}/model_entropy/{num_topics[index]}.csv'
        index = index + 1
        create_file(csv_file_result_path)
        df.to_csv(csv_file_result_path, index=False)
        pbar.close()


""" Get docs which has entropy smaller than <threshold> """
def show_docs_has_entropy_threshold(threshold, num_topics):
    folder_path = f'./turn-in/{bigram_threshold}/model_entropy/'
    dataset = pandas.read_csv(dataset_csv_path)  # Load dataset.csv file

    csv_paths = list()
    j = 0
    while j < len(num_topics):
        csv_paths.append(folder_path + str(num_topics[j]) + '.csv')
        j = j + 1

    index = 0
    pbar = tqdm.tqdm(total=len(csv_paths))
    while index < len(csv_paths):
        output_path = f'./turn-in/{bigram_threshold}/docs_entropy_less_than_0.2/{num_topics[index]}.csv'
        create_file(output_path)
        data_df = pd.read_csv(csv_paths[index])
        output_df = pd.DataFrame()
        doc_id = 0
        for entropy_value in data_df.Entropy:
            if entropy_value < threshold:  # Apply threshold
                output_df = output_df.append(pd.Series(
                    [str(doc_id), dataset['Submission_Num'][doc_id], data_df.Probabilities[doc_id], str(entropy_value),
                     dataset['Submission_Text'][doc_id]]), ignore_index=True)
            doc_id = doc_id + 1
        if output_df.empty:
            output_df = output_df.append(
                pd.Series(['Document_No', 'Submission_Num', 'Probabilities', 'Entropy', 'Submission_Text']),
                ignore_index=True)
        else:
            output_df.columns = ['Document_No', 'Submission_Num', 'Probabilities', 'Entropy', 'Submission_Text']
        output_df.to_csv(output_path, index=False, header=False)
        index = index + 1
        pbar.update(1)
    pbar.close()


""" Generate html files """
def generate_pyLDAvis_with_models(models_path, num_topics):
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)
    dict_path = './output/dictionary/all.gensim'
    dictionary = Dictionary.load(dict_path)
    tmp = dictionary[0]

    index = 0

    while index < len(models_path):
        lda_model = models.ldamodel.LdaModel.load(models_path[index])
        lda_display = pyLDAvis.gensim.prepare(
            lda_model, corpus, dictionary,
            sort_topics=True, mds='mmds'  # , R = 50
        )
        html_path = './turn-in/' + str(num_topics[index]) + '-' + '.html'
        index = index + 1
        create_file(html_path)
        pyLDAvis.save_html(lda_display, html_path)


# generate_pyLDAvis_with_models(models_path=models_path, num_topics=num_topics)


""" Calculate average number of words across all docs and standard deviation """
""" before cleaning and after cleaning process """
def calculate_mean_std_deviation(raw_dataset_csv_path, corpus_path, dictionary_path, output_path):
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)  # Bag of words
    dictionary = Dictionary.load(dictionary_path)

    raw_dataset = pandas.read_csv(raw_dataset_csv_path)
    raw_words_count = 0
    for text in raw_dataset['Submission_Text']:  # Get words count from raw data
        raw_words_count += len(text.split())
    raw_mean = raw_words_count / len(raw_dataset['Submission_Text'])

    words_count = 0
    for word_id, word_count in dictionary.cfs.items():  # Get words count from cleaned data
        words_count += word_count
    mean = words_count / len(corpus)

    raw_std_deviation = 0  # standard deviation (from raw data)
    for text in raw_dataset['Submission_Text']:
        x = len(text.split())
        raw_std_deviation += (x - raw_mean) * (x - raw_mean)
    raw_std_deviation /= len(raw_dataset['Submission_Text'])
    raw_std_deviation = math.sqrt(raw_std_deviation)

    std_deviation = 0  # standard deviation (from cleaned data)
    for doc in corpus:
        x = 0  # word count for doc
        for word in doc:
            x += word[1]
        std_deviation += (x - mean) * (x - mean)
    std_deviation /= len(corpus)
    std_deviation = math.sqrt(std_deviation)

    if not os.path.exists(output_path):
        create_file(output_path)
    rs_text = f'raw_mean = {raw_mean}\traw_stdDeviation = {raw_std_deviation}\n' \
              f'mean = {mean}\tstd_deviation = {std_deviation}'
    append_row_to_text_file(string=rs_text, path=output_path)  # Output to txt file


# calculate_mean_std_deviation(raw_dataset_csv_path='./turn-in/dataset/dataset.csv',
#                 corpus_path='./output/corpus/all-corpus.pkl',
#                 dictionary_path='./output/dictionary/all.gensim',
#                 output_path='./cal_mean_stdDeviation.txt')


def calculate_mean_std_deviation2(raw_dataset_csv_path, corpus_path, dictionary_path, output_path): # output to csv files.
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)  # Bag of words
    # dictionary = Dictionary.load(dictionary_path)
    raw_dataset = pandas.read_csv(raw_dataset_csv_path)

    raw_word_counts_list = list()
    for doc in raw_dataset['Submission_Text']:  # Get words count from raw data
        doc_word_count = len(doc.split())
        raw_word_counts_list.append(doc_word_count)
    raw_word_count_series = pandas.Series(data=raw_word_counts_list)
    raw_mean = raw_word_count_series.mean()
    raw_std_deviation = raw_word_count_series.std()

    word_counts_list = list()
    for doc in corpus:
        x = 0
        for word in doc:
            x += word[1]
        word_counts_list.append(x)
    word_counts_series = pandas.Series(data=word_counts_list)
    mean = word_counts_series.mean()
    std_deviation = word_counts_series.std()

    if not os.path.exists(output_path):
        create_file(output_path)
        df = pd.DataFrame()
        df = df.append(pd.Series(
            [bigram_threshold, mean, std_deviation]), ignore_index=True)
        df.columns = ['Bigram_Threshold', 'Mean', 'Standard_Deviation']
    else:
        df = pandas.read_csv(output_path, header=0)
        df = df.append({'Bigram_Threshold': bigram_threshold,
                        'Mean': mean,
                        'Standard_Deviation': std_deviation}, ignore_index=True)
    df.to_csv(output_path, index=False)

    # rs_text = f'mean = {mean}\tstd_deviation = {std_deviation}'
    # append_row_to_text_file(string=rs_text, path=output_path)  # Output to txt file


# calculate_mean_std_deviation2(raw_dataset_csv_path='./turn-in/dataset/dataset.csv',
#                 corpus_path='./output/corpus/all-corpus.pkl',
#                 dictionary_path='./output/dictionary/all.gensim',
#                 output_path='./cal_mean_stdDeviation.txt')


""" Get Dataset (After remove dups and empty comments) """
def change_data_set_format():
    raw_dataset_path = './data/all-comments.csv'
    df = pandas.read_csv(raw_dataset_path, header=0, encoding="ISO-8859-1")

    pandas.set_option('max_colwidth', 1000)
    submission_texts_df = df.Submission_Text
    submission_numbers_df = df.Submission_Number

    cv = CountVectorizer(stop_words=stopwords.words('english'))  # Remove some basic stop words
    combined_data = get_combined_data(submission_numbers_df, submission_texts_df)
    data_df = pandas.DataFrame.from_dict(combined_data).transpose()
    data_df.columns = ['Submission_Text']
    data_df.drop_duplicates(subset='Submission_Text', keep='first', inplace=True)  # Remove duplicated comments
    data_df = data_df.sort_index()
    submission_nums = []
    for sub_num in data_df.axes[0]:
        submission_nums.append(sub_num)
    data_df['Submission_Num'] = submission_nums

    csv_file_result_path = './dataset.csv'
    create_file(csv_file_result_path)
    data_df.to_csv(csv_file_result_path)


# change_data_set_format()


def generate_bigram_list():  # output to csv files.
    dictionary = Dictionary.load(get_paths_without_reset('all')[9])
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token
    df = pd.DataFrame()
    pbar = tqdm.tqdm(total=len(dictionary.cfs))

    for id, count in dictionary.cfs.items():
        if '_' in id2word[id]:
            df = df.append(pd.Series(
                [id2word[id], count]), ignore_index=True)
        pbar.update(1)

    output_path = f'./turn-in/{bigram_threshold}/bigram-list.csv'
    create_file(output_path)
    df.columns = ['Bigram', 'Count']
    df.sort_values(by=['Count', 'Bigram'], ascending=False, inplace=True)  # Sort columns in descending order
    df.to_csv(output_path, index=False)
    pbar.close()


def generate_topic_words():  # output to csv files.
    pbar = tqdm.tqdm(total=len(models_path))
    i = 0
    for model_path in models_path:
        df = pd.DataFrame()
        lda_model = LdaMallet.load(model_path)
        lda_model = models.wrappers.ldamallet.malletmodel2ldamodel(lda_model, iterations=iteration)
        topics_dictionary = lda_model.show_topics(num_topics=num_topics[i], num_words=30, formatted=False)

        for topic in topics_dictionary:
            topic_num = topic[0] + 1
            terms_string = ''
            for term in topic[1]:
                terms_string += term[0] + ', '
            df = df.append(pd.Series(
                [topic_num, terms_string[:-2]]), ignore_index=True)

        output_path = f'./turn-in/{bigram_threshold}/topic_terms/{num_topics[i]}.csv'
        create_file(output_path)
        df.columns = ['Topic', 'Terms']
        df.sort_values(by=['Topic'], ascending=True, inplace=True)  # Sort columns in ascending order
        df.to_csv(output_path, index=False)
        pbar.update(1)
        i = i + 1
    pbar.close()


def generate_topic_weight_terms():
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)

    pbar = tqdm.tqdm(total=len(models_path))
    i = 0
    for model_path in models_path:
        lda_model = LdaMallet.load(model_path)
        lda_model = models.wrappers.ldamallet.malletmodel2ldamodel(lda_model, iterations=iteration)
        df = topics_proportion(lda_model=lda_model, corpus=corpus, num_topics=num_topics[i])
        df.sort_values(by=['Topic'], ascending=True, inplace=True)  # Sort columns in ascending order

        output_path = f'./turn-in/{bigram_threshold}/topic_proportion_terms/{num_topics[i]}.csv'
        create_file(output_path)
        df.to_csv(output_path, index=False)
        pbar.update(1)
        i = i + 1
    pbar.close()


def generate_topic_proportion_terms():  # This calculation is based on dominant topic belonged to each doc.
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)

    pbar = tqdm.tqdm(total=len(models_path))
    i = 0
    for model_path in models_path:
        lda_model = LdaMallet.load(model_path)
        lda_model = models.wrappers.ldamallet.malletmodel2ldamodel(lda_model, iterations=iteration)

        df_dominant_topic_document = dominant_topics(lda_model=lda_model, corpus=corpus)
        # Number of Documents for Each Topic
        topic_counts = df_dominant_topic_document['Dominant_Topic'].value_counts()

        # Percentage of Documents for Each Topic
        topic_contribution = round(topic_counts / topic_counts.sum(), 4)

        # Topic Nums
        topic_nums = pd.Series(topic_contribution.index, topic_contribution.index)

        topic_terms = pd.Series()
        # Topic Terms
        topics_dictionary = lda_model.show_topics(num_topics=num_topics[i], num_words=30, formatted=False)
        for topic in topics_dictionary:
            topic_num = topic[0] + 1
            terms_string = ''
            for term in topic[1]:
                terms_string += term[0] + ', '
            topic_terms = topic_terms.append(pd.Series(terms_string[:-2], index=[topic_num * 1.0]))

        # Concatenate Column wise
        df_dominant_topics = pd.concat([topic_nums, topic_counts, topic_contribution, topic_terms], axis=1)

        # Change Column names
        df_dominant_topics.columns = ['Topic', 'Count_Documents', 'Proportion_Over_Documents', 'Terms']
        df_dominant_topics.sort_values(by=['Topic'], ascending=True, inplace=True)  # Sort columns in ascending order

        output_path = f'./turn-in/{bigram_threshold}/topic_proportion_terms/{num_topics[i]}.csv'
        create_file(output_path)
        df_dominant_topics.to_csv(output_path, index=False)
        pbar.update(1)
        i = i + 1
    pbar.close()


""" Produce to evaluate """
# coherence_scores_to_csv(models_path=models_path, num_topics=num_topics)
# generate_pyLDAvis_with_models(models_path=models_path, num_topics=num_topics)
# calculate_entropy_mallet_models()
# show_docs_has_entropy_threshold(threshold=0.2, num_topics=num_topics)
# generate_bigram_list()
# generate_topic_weight_terms()

# calculate_mean_std_deviation2(raw_dataset_csv_path='./turn-in/dataset/dataset.csv',
#                 corpus_path='./output/corpus/all-corpus.pkl',
#                 dictionary_path='./output/dictionary/all.gensim',
#                 output_path='./turn-in/Mean-StdDeviation.csv')

alarm(repeat=2)
