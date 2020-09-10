from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.tag import StanfordNERTagger
from dotenv import load_dotenv
from os.path import join, dirname
import nltk
import re
import string
import pandas
import pickle
import os

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

Stanford_modelfile = os.getenv('STANFORD_MODEL')
Stanford_NER_jarfile = os.getenv('STANFORD_NER_JAR')
ner = StanfordNERTagger(model_filename=Stanford_modelfile, path_to_jar=Stanford_NER_jarfile)

lemmatizer = nltk.wordnet.WordNetLemmatizer()
tag_dict = {"J": wn.ADJ,
            "N": wn.NOUN,
            "V": wn.VERB,
            "R": wn.ADV}


def first_round(paths, cv, data_df, stop_words):
    round1 = lambda x: first_round_cleaning(x, stop_words)
    # first round
    first_round_cleaned_data = pandas.DataFrame(data_df.Submission_Text.apply(round1))
    first_round_cleaned_data.to_pickle(paths[11])  # save as corpus for phase 2

    data_cv = cv.fit_transform(first_round_cleaned_data.Submission_Text)
    data_dtm = pandas.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.index = first_round_cleaned_data.index

    pickle.dump(cv, open(paths[3], "wb"))  # Nothing to do
    data_dtm.to_pickle(paths[2])  # save as dtm for eda to process

    return first_round_cleaned_data


def second_round(paths, first_round_cleaned_data, cv, stop_words):
    round2 = lambda x: second_round_cleaning(x, stop_words)
    # second round
    second_round_cleaned_data = pandas.DataFrame(first_round_cleaned_data.Submission_Text.apply(round2))

    data_cv = cv.fit_transform(second_round_cleaned_data.Submission_Text)
    data_dtm = pandas.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.index = second_round_cleaned_data.index

    pickle.dump(cv, open(paths[3], "wb"))  # Nothing to do
    data_dtm.to_pickle(paths[2])  # save as dtm for eda to process

    second_round_cleaned_data.to_pickle(paths[12])  # save as corpus for phase 3

    return second_round_cleaned_data


def remove_stop_words(words, stop_words):
    fil = [word for word in words if not word in stop_words]
    result = ''
    for word in fil:
        result += word + ' '
    return result.strip()


def remove_name_from_text(text):
    rs = ''
    for word, tag in ner.tag(word_tokenize(text)):
        if tag != 'PERSON':
            rs += word + ' '
    return rs.strip()


def get_postag(tag):
    # take the first letter of the tag
    # the second parameter is an "optional" in case of missing key in the dictionary
    return tag_dict.get(tag[0].upper(), None)


def lemmatize_word(key_pair):
    # key_pair: (word, pos_tag)
    tag = get_postag(key_pair[1])
    return lemmatizer.lemmatize(key_pair[0],
                                tag) if tag is not None else None  # else key_pair[0] # remove 'else key_pair[0]'


def text_list_tokenizer(ls):
    rs = []
    for i in ls:
        try:
            rs.extend(i.split(" "))
            rs.extend(i.lower().split(" "))
        except AttributeError:
            print('Null commenter.')
    return rs


def get_stop_words():  # Round 2 cleaning
    stop_words = stopwords.words('english')
    stop_words.extend([  # List of words to remove
        'im', 'just', 'dont', 'thats', 'youre', 'yeah',
        'hes', 'shes', 'theyre', 'its', 'theyll', 'bc', 'the',
        'youll', 'youd', 'isnt', 'theres', 'heres', 'would', 've',
        're', 'll'

        # Added list
        # 'go', 'goes', 'comment', 'comments'
        # 'project', 'projects', 'millennium', 'millenniums', 'millenium',
        # 'bulk', 'longviews', 'longview', 'washington', 'coal'
        # 'terminal', 'terminals', 'port', 'ports', 'cowlitz',
        # 'state', 'states', 'county', 'counties'

        # Old list
        # 'im', 'just', 'dont', 'thats', 'youre', 'yeah',
        # 'hes', 'shes', 'theyre', 'its', 'theyll', 'bc', 'the',
        # 'youll', 'youd', 'isnt', 'theres', 'heres',
        # 'would',
        # 'from', 'subject', 're', 'edu',
        # 'From', 'Subject', 'Re' 'zip',
        # 'zip code', 'Zip Code', 'Tel', 'tel', 'ad', 'PDF',
        # 'sincerely', 'Sincerely', 'SINCERELY',
        # 'adhd', 'many', 'much', 'lot', 'more',
        # 'dear', 'thank',
        # 'already', 'without',
        # 'needs', 'need', 'get', 'gets', 'like', 'likes',
    ])
    stop_words = list(map(str.lower, stop_words))
    result = [i for n, i in enumerate(stop_words) if i not in stop_words[:n]]  # Remove duplicates
    return result


def first_round_cleaning(text, stop_words):
    if not isinstance(text, str):
        return ''

    text = re.sub('[\S]+\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)[\S]*\s?', '',
                  text)  # remove URLs
    text = text.replace('\'', ' ')
    text = text.replace('\\', ' ')  # remove backslash
    text = re.sub('\[.*?\]', '', text)  # remove any brackets, parentheses... and anything inside
    text = re.sub('\w\d\w*', '', text)  # remove any word that contains digit
    text = re.sub('\S*@\S*\s?', '', text)  # remove emails
    text = re.sub(r'[^\w]', ' ', text)  # remove special chars from words which contain them. Ex: I've -> I ve
    text = re.sub('[‘’“”…]', '', text)  # remove quotations which stand alone
    text = re.sub('\S*\d\S*', "", text)  # remove words contain numbers
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # remove all punctuations
    text = re.sub('\s+', ' ', text)  # remove new lines
    text = remove_name_from_text(text)  # remove names, THIS MIGHT TAKE LOTS OF TIME
    text = text.lower()  # Adding this since gensim package input is automatically lower case
    text = ' '.join(word for word in text.split() if len(word) > 1)  # Remove word that has only 1 character.
    text = re.sub('\w*â\w*', '', text)
    text = re.sub('\w*ò\w*', '', text)
    text = re.sub('\w*õ\w*', '', text)
    text = re.sub('\w*ç\w*', '', text)
    text = re.sub('\w*ã\w*', '', text)
    text = re.sub('\w*å\w*', '', text)

    words = word_tokenize(text)
    return remove_stop_words(words, stop_words)


def second_round_cleaning(text, stop_words):
    if not isinstance(text, str):
        return ''

    words = word_tokenize(text)
    words_tags = nltk.pos_tag(words)  # returns a list of tuples: (word, tagString) like ('And', 'CC')
    lemmatized_words = [lemmatize_word(key_pair) for key_pair in words_tags]
    fil = list(filter(None, lemmatized_words))  # lemmatizing words

    result = []
    for word in fil:
        result.append(word)
    return remove_stop_words(result, stop_words)

    # result = ''
    # for word in fil:
    #     result += word + ' '
    # print('Cleaning...')
    # return result.strip()
