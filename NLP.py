import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from tabulate import tabulate

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


def removePunct(Corpus):
    ponctuationRemoved = Corpus['body'].str.replace(r'[^\w\s]+', '')
    return ponctuationRemoved


def toLowercase(Corpus):
    lowerCased = Corpus['punctRemoved'].str.lower()
    return lowerCased


def removeStopWords(Corpus):
    stop = stopwords.words('english')
    removedStopWords = Corpus['lowerCased'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return removedStopWords


def TokenizeWords(Corpus):
    TokenizedWords = Corpus['removedStopWords'].apply(nltk.word_tokenize)
    return TokenizedWords


def LemmatizeWords(Corpus):
    LemmatizedWords = Corpus['TokenizedWords'].apply(lambda x: [WordNetLemmatizer().lemmatize(y) for y in x])
    return LemmatizedWords


def create_bigram_dic(bigram_corpus):

    bigram_dic = {}
    for bigram in bigram_corpus:
        if bigram not in bigram_dic:
            bigram_dic[bigram] = 1
        else:
            bigram_dic[bigram] += 1
    return bigram_dic


def create_unigram_dic(unigram_corpus):
    unigram_dic = {}
    for unigram in unigram_corpus:
        if unigram not in unigram_dic:
            unigram_dic[unigram] = 1
        else:
            unigram_dic[unigram] += 1
    return unigram_dic


def calculate_ham_bigram_probability(w1, w2):
    try:
        no_w1 = ham_unigram_dic[w1]
    except KeyError:
        no_w1 = 0

    try:
        w1_w2 = ham_bigram_dic[w1, w2]
    except KeyError:
        w1_w2 = 0

    V = len(ham_unigram_dic)
    p = (w1_w2 + 1) / (no_w1 + V)
    return p


def calculate_spam_bigram_probability(w1, w2):
    try:
        no_w1 = spam_unigram_dic[w1]
    except KeyError:
        no_w1 = 0

    try:
        w1_w2 = spam_bigram_dic[w1, w2]
    except KeyError:
        w1_w2 = 0

    V = len(spam_unigram_dic)
    p = (w1_w2 + 1) / (no_w1 + V)
    return p


def preprocess(message):
    message = message.lower()
    message = re.sub(r'[^\w\s]', '', message)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(message)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    message = " ".join(filtered_sentence)
    word_tokens = word_tokenize(message)

    LemmatizedWords = []
    for ms in word_tokens:
        LemmatizedWords.append(WordNetLemmatizer().lemmatize(ms))

    bigram = list(ngrams(LemmatizedWords, 2))
    return bigram


def calculate_ham_laplace_smoothing_prob(bigrams):
    p = 1
    for bigram in bigrams:
        p = p * calculate_ham_bigram_probability(bigram[0], bigram[1])
    return p


def calculate_spam_laplace_smoothing_prob(bigrams):
    p = 1
    for bigram in bigrams:
        p = p * calculate_spam_bigram_probability(bigram[0], bigram[1])
    return p


def classify_msg_type(ham_prob, spam_prob):
    if ham_prob >= spam_prob:
        return "Ham Message"
    else:
        return "Spam Message"


Corpus = pd.read_csv("SMSSpamCollection.tsv", sep="\t", header=None)
Corpus.columns = ["lable", "body"]
Corpus.head()

Corpus['punctRemoved'] = removePunct(Corpus)
Corpus['lowerCased'] = toLowercase(Corpus)
Corpus['removedStopWords'] = removeStopWords(Corpus)
Corpus['TokenizedWords'] = TokenizeWords(Corpus)
Corpus['LemmatizedWords'] = LemmatizeWords(Corpus)

Corpus['Bigrams'] = Corpus['LemmatizedWords'].apply(lambda x: list(ngrams(x, 2)))
bigram_data_frame = Corpus.groupby('lable').agg({'Bigrams': 'sum'})
unigram_data_frame = Corpus.groupby('lable').agg({'LemmatizedWords': 'sum'})

ham_unigram_corpus = unigram_data_frame.iat[0, 0]
spam_unigram_corpus = unigram_data_frame.iat[1, 0]

ham_bigram_corpus = bigram_data_frame.iat[0, 0]
spam_bigram_corpus = bigram_data_frame.iat[1, 0]

spam_unigram_dic = create_unigram_dic(spam_unigram_corpus)
ham_unigram_dic = create_unigram_dic(ham_unigram_corpus)

spam_bigram_dic = create_bigram_dic(spam_bigram_corpus)
ham_bigram_dic = create_bigram_dic(ham_bigram_corpus)

message1 = "Sorry, ..use your brain dear"
message2 = "SIX chances to win CASH."

message1_bigram = preprocess(message1)
message2_bigram = preprocess(message2)

msg1_ham_prob = calculate_ham_laplace_smoothing_prob(message1_bigram)
msg1_spam_prob = calculate_spam_laplace_smoothing_prob(message1_bigram)

msg2_ham_prob = calculate_ham_laplace_smoothing_prob(message2_bigram)
msg2_spam_prob = calculate_spam_laplace_smoothing_prob(message2_bigram)

print(tabulate([[message1, msg1_ham_prob, msg1_spam_prob, classify_msg_type(msg1_ham_prob, msg1_spam_prob)],
                [message2, msg2_ham_prob, msg2_spam_prob, classify_msg_type(msg2_ham_prob, msg2_spam_prob)]],
               headers=['message', 'ham probability', 'spam probability', 'message type']))

