import string
import collections
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import textblob
import matplotlib as matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import emoji
import pandas as pd
import sklearn.metrics as metrics
from nltk.corpus import stopwords
from nltk.corpus import words as nltk_words
from nltk.stem import PorterStemmer
import re
import sklearn.cluster as cluster
from sklearn.neural_network import MLPClassifier
from wordcloud import WordCloud
import nltk.sentiment.vader as vaderSentimentScore
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
dictionary = dict.fromkeys(nltk_words.words(), None)

positive_count = 0
negative_count = 0
neutral_count = 0


# i=0
def preliminary_analysis(comments):
    global positive_count, negative_count, neutral_count
    unique_words = set()
    total_word = 0
    for i in range(0, len(comments)):
        sentence = comments.iloc[i]["text"]
        sentiment_score = comments.iloc[i]["score"]
        count = sentence.count(":D")
        if sentiment_score == 0:
            neutral_count += count
        elif sentiment_score == -1:
            negative_count += count
        else:
            positive_count += count

        number_of_words = len(sentence.split())
        total_word += number_of_words
        words = sentence.split()
        for word in words:
            unique_words.add(word)

    print("average words: ", total_word / len(comments))
    print("total words: ", total_word)
    print("unique_words: ", len(unique_words))
    print("positive class count :D : ", positive_count)
    print("negative class count :D : ", negative_count)
    print("neutral class count :D : ", neutral_count)


def plot(comments):
    global positive_count, negative_count, neutral_count
    positive_unique_words = set()
    negative_unique_words = set()
    neutral_unique_words = set()
    unique_words = set()
    for i in range(0, len(comments)):
        sentence = comments.iloc[i]["text"]
        sentiment_score = comments.iloc[i]["score"]
        words = sentence.split()

        for word in words:
            unique_words.add(word)
            if sentiment_score == 1 and not (word in positive_unique_words):
                positive_unique_words.add(word)
            elif sentiment_score == -1 and not (word in negative_unique_words):
                negative_unique_words.add(word)
            elif sentiment_score == 0 and not (word in neutral_unique_words):
                neutral_unique_words.add(word)

    print("positive words: ", len(positive_unique_words))
    print("negative words: ", len(negative_unique_words))
    print("neutral words: ", len(neutral_unique_words))
    a = len(positive_unique_words)
    b = len(negative_unique_words)
    c = len(neutral_unique_words)
    names = ["positive", "negative", "neutral"]
    values = [a, b, c]
    plt.bar(names, values)

    plt.show()


def most_frequent_words(comments):
    positive_text = ""
    negative_text = ""
    neutral_text = ""
    for i in range(0, len(comments)):
        sentence = comments.iloc[i]["text"]
        sentiment_score = comments.iloc[i]["score"]
        if sentiment_score == 1:
            positive_text += sentence + " "
        elif sentiment_score == -1:
            negative_text += sentence + " "
        else:
            neutral_text += sentence + " "

    positive_counter = collections.Counter(positive_text.split())
    negative_counter = collections.Counter(negative_text.split())
    neutral_counter = collections.Counter(neutral_text.split())
    print("Positive top 10: ", positive_counter.most_common(10))
    print("Negatve top 10: ", negative_counter.most_common(10))
    print("Neutral top 10: ", neutral_counter.most_common(10))


def preprocess(comments):
    stop_words = stopwords.words('english')
    stemmer = PorterStemmer()
    for i in range(0, len(comments)):
        sentence = comments.iloc[i]["text"]
        sentence = sentence.lower()
        sentence = emoji.demojize(sentence)
        sentence.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        sentence = re.sub(r'[^\w\s]', '', sentence)  # remove white space
        sentence = re.sub(r'[0-9]+', '', sentence)  # remove number
        sentence = sentence.replace(' na ', '')
        sentence = sentence.replace(' er ', '')
        sentence = sentence.replace(' ta ', '')
        #sentence = sentence.replace('u', '')
        sentence = sentence.replace(' chai ', '')
        sentence = sentence.replace(' r ', '')
        #sentence = sentence.replace('yo', '')
        sentence = sentence.replace(' ae ', '')
        sentence = sentence.replace(' bo ', '')
        sentence = sentence.replace(' x ', '')
        sentence = sentence.replace(' p ', '')
        sentence = sentence.replace(' k ', '')
        sentence = sentence.replace('song', '')
        sentence = sentence.replace('video', '')
        sentence = sentence.replace(' u ', '')

        #sentence = sentence.lstrip('na')
        words = sentence.split()
        new_sentence = ""
        for word in words:
            if word not in stop_words:
                word = stemmer.stem(word)  # stemming
                if is_english_word(word):
                    new_sentence += word + " "


        comments.iloc[i, comments.columns.get_loc('text')] = new_sentence
    return comments

def is_english_word(word):
    try:
        x = dictionary[word]
        return True
    except KeyError:
        return False

def get_text_from_data(tweets):

    return ' '.join(tweets['text'])

def generate_wordcloud(comments):
    positive_comments = comments.loc[comments['score'] == 1]
    negative_comments = comments.loc[comments['score'] == -1]
    neutral_comments = comments.loc[comments['score'] == 0]

    wordcloud = WordCloud().generate(get_text_from_data(comments))
    plt.title('Overall')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

    positive_wordcloud = WordCloud().generate(get_text_from_data(positive_comments))
    plt.title('Positive')
    plt.imshow(positive_wordcloud)
    plt.axis("off")
    plt.show()

    negative_wordcloud  = WordCloud().generate(get_text_from_data(negative_comments))
    plt.title('Negative')
    plt.imshow(negative_wordcloud)
    plt.axis("off")
    plt.show()

    neutral_wordcloud = WordCloud(relative_scaling=1.0).generate(get_text_from_data(neutral_comments))
    plt.title('Neutral')
    plt.imshow(neutral_wordcloud)
    plt.axis("off")
    plt.show()

def senti(sentence):
    return textblob.TextBlob(sentence).polarity
    #return vaderSentimentScore.SentimentIntensityAnalyzer().polarity_scores(sentence).get('compound')


def sentiment_stats(comments):
    positive = []
    negative = []
    neutral = []
    true_class = []
    predicted_class = []
    for i in range(0, len(comments)):
        sentence = comments.iloc[i]["text"]
        sentiment_score = comments.iloc[i]["score"]
        sentiment_value = senti(sentence)
        if(sentiment_value>0):
            predicted_class.append('Positive')
        elif sentiment_value<0:
            predicted_class.append('Negative')
        else:
            predicted_class.append('Neutral')
        if sentiment_score == 1:
            positive.append(sentiment_value)
            true_class.append('Positive')
        elif sentiment_score == -1:
            negative.append(sentiment_value)
            true_class.append('Negative')
        elif sentiment_score == 0:
            neutral.append(sentiment_value)
            true_class.append('Neutral')
    my_confusion_matrix = confusion_matrix(true_class, predicted_class, labels=["Positive", "Negative", "Neutral"])
    print(my_confusion_matrix)

    plt.hist(neutral, 100)
    plt.xlabel('sentiment score')
    plt.ylabel('neutral comments')
    plt.show()

    plt.hist(positive, 100)
    plt.xlabel('sentiment score')
    plt.ylabel('positive comments')
    plt.show()

    plt.hist(negative, 100)
    plt.xlabel('sentiment score')
    plt.ylabel('negative comments')
    plt.show()

def tf_idf_step(comments):



    Y = comments.score

    type_of_comment = comments.domain.values

    positive_tweets = comments.loc[comments['score'] == 1]
    negative_tweets = comments.loc[comments['score'] == -1]
    neutral_tweets = comments.loc[comments['score'] == 0]
    corpus = comments.loc[:, "text"]
    vectorizer = TfidfVectorizer(ngram_range=(1,1))
    X = vectorizer.fit_transform(corpus)

    features = vectorizer.get_feature_names()
    print("Size of corpus: ", len(features))
    positive_rows = []
    negative_rows = []
    neutral_rows = []

    total_unigram_count = 0
    positive_unigram_count = 0
    negative_unigram_count = 0
    neutral_unigram_count = 0

    positive_common_words = []
    negative_common_words = []
    neutral_common_words = []
    total_common_words = []
    for i in range(0, X.shape[0]):
        # print(i)
        sentiment = comments.iloc[i]["score"]
        row = X[i]
        if sentiment == 1:
            positive_rows.append(row)
            for index in row.indices:
                positive_word = features[index]
                positive_common_words.append(positive_word)
            positive_unigram_count += len(row.indices)
        elif sentiment == -1:
            negative_rows.append(row)
            for index in row.indices:
                negative_word = features[index]
                negative_common_words.append(negative_word)
            negative_unigram_count += len(row.indices)
        else:
            neutral_rows.append(row)
            for index in row.indices:
                neutral_word = features[index]
                neutral_common_words.append(neutral_word)
            neutral_unigram_count += len(row.indices)
        for index in row.indices:
            word = features[index]
            total_common_words.append(word)
        total_unigram_count += len(row.indices)


    total_counter = collections.Counter(total_common_words)
    positive_counter = collections.Counter(positive_common_words)
    negative_counter = collections.Counter(negative_common_words)
    neutral_counter = collections.Counter(neutral_common_words)
    '''  
    print("Size of positive unigrams: ", len(positive_counter))
    print("Size of negative unigrams: ", len(negative_counter))
    print("Size of neutral unigrams: ", len(neutral_counter))

    print("Average unigram count in positive class: ", positive_unigram_count * 1.0 / len(positive_tweets))
    print("Average unigram count in negative class: ", negative_unigram_count * 1.0 / len(negative_tweets))
    print("Average unigram count in neutral class: ", neutral_unigram_count * 1.0 / len(neutral_tweets))
    print("Average unigram count in total: ", total_unigram_count * 1.0 / len(comments))

    positive_most_common_words = positive_counter.most_common(10)
    print("Positive common bigrams: ", positive_most_common_words)
    negative_most_common_words = negative_counter.most_common(10)
    print("Negative common bigrams: ", negative_most_common_words)
    neutral_most_common_words = neutral_counter.most_common(10)
    print("Neutral common bigrams: ", neutral_most_common_words)

    print("Total common unigrams: ", total_counter.most_common(10))
    '''
    # print(positive_columns)


    return X, Y


def perform_K_means_clustering(X,Y):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    for i in range (0, len(X)):
        print(str(kmeans.labels_[i])+" "+ Y[i])




def perform_steps_upto_feature_extraction():
    # read csv file
    table = pd.read_csv("Sentiment.csv", sep=';')

    # Take only EN rows
    comments = table.loc[table['lan'] == 'EN']

    # Number of comments
    number = len(comments)
    print("number of youtube comments: ", number)

    words_count = comments.count().sum()
    print("words: ", words_count)

    # number of words(total,average,unique) in each comments
    # preliminary_analysis(comments)

    # drop 2 columns
    comments = comments.drop(columns=['id', 'lan', 'label'])

    # mapping scores
    sentiment_mapper = {"score": {-2: -1, 2: 1}, "domain": {"funny_video": 0, "music_video": 1, "review_video": 2,
                                                            "drama_video": 3, "sports_video": 4, "talkshow_video": 5,
                                                            "report_video": 6, "news_video": 7}}
    # print(pd.value_counts(comments.domain))
    comments.replace(sentiment_mapper, inplace=True)
    #sentiment_stats(comments)
    #preliminary_analysis(comments)
    # plot(comments)
    preprocess(comments)
    #generate_wordcloud(comments)
    preliminary_analysis(comments)
    #plot(comments)
    #most_frequent_words(comments)

    # # print(comments)
    # print(comments.iloc[500][:])
    X, Y = tf_idf_step(comments)
    return X,Y


X,Y = perform_steps_upto_feature_extraction()


from sklearn import svm

# Create the model with 100 trees
model = clf = MLPClassifier(solver='lbfgs',activation='relu', alpha=1e-5,
                    hidden_layer_sizes=(5,), random_state=1)
from sklearn.naive_bayes import MultinomialNB

model = clf = MultinomialNB()
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42)
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print("Accuracy: %.2f%%" % (result*100.0))
#perform_K_means_clustering(X, Y)


k_means = cluster.KMeans(n_clusters=3)
k_means.fit(X_train)
#print(k_means.labels_[:])
#print(y_train[:])
#print(cross_val_score(clf, X, Y, cv=5))
score = metrics.accuracy_score(y_test,k_means.predict(X_test))
print('Accuracy:{0:f}'.format(score))

k_means.predict(X_test)
print(k_means.labels_)
print(y_test)