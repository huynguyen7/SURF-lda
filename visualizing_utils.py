import pickle
import matplotlib.pyplot as plt
import pyLDAvis
from gensim import models
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.collocations import BigramCollocationFinder
from main_utils import append_row_to_text_file, append_row_to_csv
from paths import create_file
from pyLDAvis import gensim

def data_to_words(data):
    rs = []
    for text in data.Submission_Text:
        rs.extend(word_tokenize(text))
    return rs


def get_words_frequency(data, path):
    fdist = FreqDist()
    for text in data.Submission_Text:
        for word in word_tokenize(text):
            fdist[word] += 1
    for word in fdist.most_common():
        append_row_to_text_file((word[0] + ": " + str(word[1])), path)
    return fdist


def get_words_bigrams_frequency(docs, path):
    fdist = FreqDist()
    for doc in docs:
        for word in doc:
            fdist[word] += 1
    for word in fdist.most_common():
        append_row_to_text_file((word[0] + ": " + str(word[1])), path)


def visualize_LDA(paths, num_topics, passes, alpha=None, beta=None):
    dictionary = Dictionary.load(paths[9])
    with open(paths[1], 'rb') as f:
        corpus = pickle.load(f)
    temp = dictionary[0]  # This is only to "load" the dictionary.

    if alpha is not None and beta is not None:
        model_path = paths[16] + str(alpha) + '_alpha_' + str(beta) + '_beta_' + str(num_topics) + '-' + str(passes) + '.gensim'
        html_path = paths[15] + str(alpha) + '_alpha_' + str(beta) + '_beta_' + str(num_topics) + '-' + str(passes) + '.html'
    else:
        model_path = paths[16] + str(num_topics) + '-' + str(passes) + '.gensim'
        html_path = paths[15] + str(num_topics) + '-' + str(passes) + '.html'
    lda_model = models.ldamodel.LdaModel.load(model_path)
    lda_display = pyLDAvis.gensim.prepare(
        lda_model, corpus,
        dictionary, sort_topics=True, mds='mmds'
    )

    create_file(html_path)
    pyLDAvis.save_html(lda_display, html_path)


def visualize_LDA_mallet(paths, num_topics, passes, alpha=None, beta=None):
    dictionary = Dictionary.load(paths[9])
    with open(paths[1], 'rb') as f:
        corpus = pickle.load(f)
    temp = dictionary[0]  # This is only to "load" the dictionary.

    if alpha is not None and beta is not None:
        model_path = paths[16] + str(alpha) + '_alpha_' + str(beta) + '_beta_' + str(num_topics) + '-' + str(passes) + '.gensim'
        html_path = paths[15] + str(alpha) + '_alpha_' + str(beta) + '_beta_' + str(num_topics) + '-' + str(passes) + '.html'
    else:
        model_path = paths[16] + str(num_topics) + '-' + str(passes) + '.gensim'
        html_path = paths[15] + str(num_topics) + '-' + str(passes) + '.html'

    mallet_model = models.ldamodel.LdaModel.load(model_path)
    mallet_lda_model = models.wrappers.ldamallet.malletmodel2ldamodel(mallet_model, iterations=passes)

    lda_display = pyLDAvis.gensim.prepare(
        mallet_lda_model, corpus,
        dictionary, sort_topics=False, mds='mmds'
    )

    create_file(html_path)
    pyLDAvis.save_html(lda_display, html_path)


def show_bigrams_freq(data, path):
    words = data_to_words(data)
    finder = BigramCollocationFinder.from_words(words=words)
    finder.apply_freq_filter(min_freq=2)  # At least 10
    sorted_dict = {k: v for k, v in sorted(finder.ngram_fd.items(), key=lambda item: item[1], reverse=True)}

    for k, v in sorted_dict.items():
        append_row_to_csv(
            file=path,
            row=str(k) + ' = ' + str(v)
        )

def graph_coherence_scores_num_topics(coherence_values, graph_path, fr, to, step):
    x = range(fr, to, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    create_file(graph_path)
    plt.savefig(graph_path)


def graph_coherence_scores_alpha(coherence_values, graph_path, fr, to, step):  # This method is not being used
    plt.plot(coherence_values)
    plt.xlabel("Alpha")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    create_file(graph_path)
    plt.savefig(graph_path)


def get_words_frequency2(dtm, path):
    word_counts = {}
    for word in dtm.columns.tolist():
        total = dtm[word].sum()
        word_counts.update({word: total})

    sorted_dict = {k: v for k, v in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)}
    for k, v in sorted_dict.items():
        append_row_to_text_file(str(k + ": " + str(v)), path)
        # print(k + ": " + str(v))
    return word_counts


def print_topic_coherence_to_text_file(text_file_path, num_topics, lda_model, corpus, path_dict):
    create_file(text_file_path)
    # top_topics = lda_model.top_topics(corpus)
    # avg_topic_coherence = sum([topic[1] for topic in top_topics]) / num_topics
    # append_row_to_text_file(
    #     str('Average topic coherence: %.9f\n' % avg_topic_coherence),
    #     text_file_path
    # )

    dictionary = Dictionary.load(path_dict)
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token
    coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, dictionary=id2word, coherence='u_mass')
    coherence_lda = coherence_model_lda.get_coherence()
    append_row_to_text_file(str('Coherence Score: %.9f\n' % coherence_lda),
                            text_file_path)


def print_perplexity_to_text_file(text_file_path, lda_model, corpus):
    append_row_to_text_file(str('Log Perplexity: %.9f\n' % lda_model.log_perplexity(corpus)),
                            text_file_path)


def print_topics_to_text_file(lda_model, text_file_path, num_words):
    for topic in lda_model.show_topics(num_words=num_words):
        rs = str('Topic: ' + str(topic[0] + 1) + '\n'
                 + str(topic[1]))
        append_row_to_text_file(rs, text_file_path)


def get_top_10_words_each_comment(tdm, path):
    top_10_words = {}
    for word in tdm.columns:
        top = tdm[word].sort_values(ascending=False).head(10)
        top_10_words[word] = list(zip(top.index, top.values))
        rs = str(
            word + ':\n'
            + str(top_10_words[word])
        )
        append_row_to_text_file(rs, path)
    # print(top_10_words)
    return top_10_words


def graphs_convergence_perplexity_coherence(all_metrics, num_topics, iterations, passes):
    for metric in ['Coherence', 'Perplexity', 'Convergence', 'docs_converged']:
        fig, axs = plt.subplots(1, 1, figsize=(15, 5))

        # Each plot to show results for all models with the same num_topics
        for i, topic_number in enumerate([num_topics]):
            filtered_topics = all_metrics[all_metrics['topics'] == topic_number]
            for label, df in filtered_topics.groupby(['iterations']):
                print(label)
                df.plot(x='pass_num', y=metric, ax=axs, label=label)

            axs.set_xlabel(f"Pass number")
            axs.legend()
            axs.set_ylim([all_metrics[metric].min(), all_metrics[metric].max()])
        if metric == 'docs_converged':
            fig.suptitle('Documents converged', fontsize=20)
        else:
            fig.suptitle(metric, fontsize=20)
        # save plot
        if metric != 'docs_converged':
            save_path = f"./evaluations/graph_scores/{num_topics}_topics_{iterations[-1]}_iterations_{passes}_passes/graphs/{metric}.png"
        else:
            save_path = f"./evaluations/graph_scores/{num_topics}_topics_{iterations[-1]}_iterations_{passes}_passes/graphs/Docs_Converged.png"
        fig.savefig(save_path)
