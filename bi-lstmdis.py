# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 10:34:09 2022

@author: HP
"""
import itertools
import pandas as pd
import numpy as np
import re
import seaborn as sns
from matplotlib import style
style.use('ggplot')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk.corpus
from nltk.corpus import stopwords

# Text Vectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Word Cloud
from wordcloud import WordCloud

stop_words = set(stopwords.words('english'))
from wordcloud import wordcloud
import matplotlib.pyplot as plt
import random
import nltk
nltk.download("stopwords")
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import numpy as np
#import tensorflow as tf
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize

np.random.seed(1)

import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, classification_report, log_loss

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, Flatten
#from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping





#print(os.listdir('../input'))
##
df = pd.read_csv("C:/Users/HP/realvascu77.csv")

df.info()

#df3_merged.drop_duplicates(inplace=True)
df.head()

#preprocessing for EDA

def Text_Cleaning(Text):
  # Lowercase the texts
  Text = Text.lower()

  # Cleaning punctuations in the text
def data_processing(text):
    text = text.lower() #lower text conversion
    text = re.sub(r"https\s+|www\s+", '',text,flags=re.MULTILINE)
    text = re.sub(r"@([a-zA-Z0-9_]{1,50})",'',text)
    text = re.sub(r'\@w+|\#','',text)
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub(r'andrewt','',text)
    
    return text

def Text_Processing(Text):
  Processed_Text = list()
  Lemmatizer = WordNetLemmatizer()

  # Tokens of Words
  Tokens = nltk.word_tokenize(Text)

  # Removing Stopwords and Lemmatizing Words
  # To reduce noises in our dataset, also to keep it simple and still 
  # powerful, we will only omit the word `not` from the list of stopwords

  for word in Tokens:
    if word not in stop_words:
      Processed_Text.append(Lemmatizer.lemmatize(word))

  return(" ".join(Processed_Text))

  return Text

#applying function to data

df["text"] = df["text"].apply(lambda text: data_processing(text))
df["text"] = df["text"].apply(lambda Text: Text_Processing(Text))


analyser = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score["compound"]

df["polarity"]=df["text"].apply(sentiment_analyzer_scores)

# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
sns.set(style="darkgrid")
hist_df = df["polarity"]
sns

sns.histplot(data=hist_df, x=df["polarity"])
plt.show()


def sentiment_col(y):
    if y >= 0.05 :
        return 'positive'
    elif y > -0.05 <0.05 :
        return 'neutral'
    elif y <= 0.05:
        return 'negative'
df['polarity'] = df['polarity'].apply(sentiment_col)

dl_df = df.copy()
df2 = df.groupby('polarity').size().reset_index(name='coun')
n = df2['polarity'].unique().__len__()+1
all_colors = list(plt.cm.colors.cnames.keys())
random.seed(1000)
c = random.choices(all_colors, k=n)

# Plot Bars
plt.figure(figsize=(10,5), dpi= 200)
plt.bar(df2['polarity'], df2['coun'], color=c, width=.5)
for i, val in enumerate(df2['coun'].values):
    plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':25})

# Decoration
plt.gca().set_xticklabels(df2['polarity'], rotation=60, horizontalalignment= 'right')
plt.title("Vasculitis Sentiment", fontsize=22)
plt.ylabel('Text', fontsize=22)
plt.ylim(0, 1700)
plt.show()



#EDA

df["length"] = df["text"].astype(str).apply(len)
df["length"].plot(kind = "hist", bins = 40, edgecolor = "blue", linewidth = 1, color = "orange", figsize = (10,5))
plt.title("Length of Text", color = "blue", pad = 20)
plt.xlabel("Length", labelpad = 15, color = "black")
plt.ylabel("Amount of Text", labelpad = 20, color = "green")

plt.show()

df["word_counts"] = df["text"].apply(lambda x: len(str(x).split()))
df["word_counts"].plot(kind = "hist", bins = 40, edgecolor = "blue", linewidth = 1, color = "orange", figsize = (10,5))
plt.title("Length of Word", color = "blue", pad = 20)
plt.xlabel("Length", labelpad = 15, color = "black")
plt.ylabel("Amount of Word", labelpad = 20, color = "green")

plt.show()

#NGRAM ANALYSIS

def Gram_Analysis(Corpus, Gram, N):
  # Vectorizer
  Vectorizer = CountVectorizer(stop_words = stop_words, ngram_range=(Gram,Gram))

  # N-Grams Matrix
  ngrams = Vectorizer.fit_transform(Corpus)

  # N-Grams Frequency
  Count = ngrams.sum(axis=0)

  # List of Words
  words = [(word, Count[0, idx]) for word, idx in Vectorizer.vocabulary_.items()]

  # Sort Descending With Key = Count
  words = sorted(words, key = lambda x:x[1], reverse = True)

  return words[:N]

# Use dropna() so the base DataFrame is not affected
Positive = df[df["polarity"] == "positive"].dropna()
Neutral = df[df["polarity"] == "neutral"].dropna()
Negative = df[df["polarity"] == "negative"].dropna()



#WORD CLOUD AND FREQUENCY DISTRIBUTION
from nltk.tokenize import word_tokenize
pos_tweet = df[df.polarity == 'positive']
text = ' '.join([word for word in pos_tweet['text']])
corpus = text
sentences = nltk.sent_tokenize(corpus)

print(type(sentences))

sentence_tokens = ""
for sentence in sentences:
    sentence_tokens += sentence
    


#word tokenization

words=nltk.word_tokenize(sentence_tokens)
print(words)
for word in words:
    print(word)
    

from nltk.corpus import stopwords
print(stopwords.words('english'))

#stop word removal
stop_words = set(stopwords.words('english'))
filtered_words=[]

for w in words:
    if w not in stop_words:
        filtered_words.append(w)

print('/n With stop words:', words)
print('/n After removing stop words:', filtered_words)


#finding the frequency distribution of words
frequency_dist = nltk.FreqDist(filtered_words)

#SORTING THE FREQUENCY DISTRIBUTION
sorted(frequency_dist,key=frequency_dist.__getitem__,reverse=True)[0:30]

#Keeping only the large words(more than 3 characters)
large_words = dict([(k,v) for k,v in frequency_dist.items() if len (k)>4])

frequency_dist = nltk.FreqDist(large_words)
frequency_dist.plot(30, cumulative=False)

#Visualising the distribution of words using matplotlib and wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt


wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="white").generate_from_frequencies(frequency_dist)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#Negative Frequency Distribution and Word cloud


neg_tweet = df[df.polarity == 'negative']
text = ' '.join([word for word in neg_tweet['text']])
corpus = text
sentences = nltk.sent_tokenize(corpus)

print(type(sentences))

sentence_tokens = ""
for sentence in sentences:
    sentence_tokens += sentence
    


#word tokenization

words=nltk.word_tokenize(sentence_tokens)
print(words)
for word in words:
    print(word)
    

from nltk.corpus import stopwords
print(stopwords.words('english'))

#stop word removal
stop_words = set(stopwords.words('english'))
filtered_words=[]

for w in words:
    if w not in stop_words:
        filtered_words.append(w)

print('/n With stop words:', words)
print('/n After removing stop words:', filtered_words)


#finding the frequency distribution of words
frequency_dist = nltk.FreqDist(filtered_words)

#SORTING THE FREQUENCY DISTRIBUTION
sorted(frequency_dist,key=frequency_dist.__getitem__,reverse=True)[0:30]

#Keeping only the large words(more than 3 characters)
large_words = dict([(k,v) for k,v in frequency_dist.items() if len (k)>4])

frequency_dist = nltk.FreqDist(large_words)
frequency_dist.plot(30, cumulative=False)

#Visualising the distribution of words using matplotlib and wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt


wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="black").generate_from_frequencies(frequency_dist)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


train_text, test_text, train_cat, test_cat = train_test_split(dl_df['text'],dl_df['polarity'], test_size = 0.20, random_state = 42)

#Machine Learning classification
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 1744
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 254
# This is fixed.
EMBEDDING_DIM = 100
#Tokenization for deep learning
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(dl_df['text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#Truncate and pad the input sequences so that they are all in the same length for modeling
X = tokenizer.texts_to_sequences(dl_df['text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

#Converting categorical labels to numbers
Y = pd.get_dummies(dl_df['polarity']).values
print('Shape of label tensor:', Y.shape)

#Splitting of data to 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

#lstm model
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 10
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();

#test new data with new complaint
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show();


# Here's how to generate a prediction on individual examples
encoder = LabelEncoder()
encoder.fit(train_cat)
text_labels = encoder.classes_ 

for i in range(80):
    prediction = model.predict(np.array([X_test[i]]))
    predicted_label = text_labels[np.argmax(prediction)]
    print(test_text.iloc[i][:400], "...")
    print('Actual label:' + test_cat.iloc[i])
    print("Predicted label: " + predicted_label + "\n")
    

    #    This function prints and plots the confusion matrix.

    #    Normalization can be applied by setting `normalize=True`.

new_complaint = ['']
seq = tokenizer.texts_to_sequences(new_complaint)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
labels = ['Negative','Neutral','Positive']
print(pred, labels[np.argmax(pred)])

# plotting history
from matplotlib import pyplot
pyplot.xlabel('Epoch')
pyplot.ylabel('Loss')
pyplot.plot(history.history['loss'],     label='train')
pyplot.plot(history.history['val_loss'], label='eval')
pyplot.legend()
pyplot.show()

# evaluating the model

from sklearn.metrics import confusion_matrix, classification_report


Predict = model.predict(X_test)


Y_pred = np.argmax(Predict, axis=1)
Y_true = np.argmax(Y_test,  axis=1)



#evaluation
import seaborn as sns
import matplotlib.pyplot as plt
cf_matrix = confusion_matrix(Y_true, Y_pred)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
            fmt='.2%', cmap='Blues')

plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.title('Confusion matrix')
plt.show()

cf_matrix = confusion_matrix(Y_true, Y_pred)
sns.heatmap(cf_matrix, annot=True)



print("\n Classification Report:")
target_classes = ['No event 0 (-)', 'Event 1 (+)', 'Event 2 (+)']
print(classification_report(Y_true, Y_pred, target_names=target_classes))
print(classification_report(Y_true, Y_pred))
