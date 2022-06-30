# Imports
import numpy as np  
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import string
%pip install pyLDAvis
import nltk
import gensim
import pyLDAvis.gensim_models 

# You'll probably need to download some nltk packages
nltk.download()

cookiemonster_url = 'https://drive.google.com/uc?export=download&id=10Av-HVklZA4Su3TcvSpxU6nS8At9mJ05'

#use stem, parts of speech, sentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer() 
stopwords = nltk.corpus.stopwords.words('english')
punc_stop = ["'",".","...","'''","://",".....","...."]
twitter_stop = ["twitter","com","pic","http","https","www","status","bit","ly"]
# Get a tokenizer
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = nltk.corpus.stopwords.words('english')

# Create p_stemmer of class PorterStemmer
p_stemmer = nltk.stem.porter.PorterStemmer()

df = pd.read_excel(cookiemonster_url)

df['tweet-length'] = [len(t) for t in df['tweet-text']]
print(df.head(5))
df['tweet-length'].hist()

plt.figure(figsize = (10,8))
df['sentiment'] = df['tweet-text'].map(lambda x: sid.polarity_scores(x)["compound"])
print(df[['tweet-text','sentiment']].head(5))
df['sentiment'].hist()
plt.xlabel('Score')
plt.ylabel('Compound')
plt.title('Cookie Monster Polarity Scores')

df['sentiment'] = df['tweet-text'].map(lambda x: sid.polarity_scores(x)["compound"])

print(df[['tweet-text','sentiment']].head(10))


new_df = df.loc[df['sentiment'] < 0] 
new_df[['tweet-text','sentiment']].head(10)

df['tweet-tokens'] = df['tweet-text'].map(lambda x: nltk.wordpunct_tokenize(x))
df[['tweet-text','tweet-tokens']].head(5)

def remove_stop_punct(token_list):
    return [t.lower() for t in token_list if (t not in string.punctuation and t.lower() not in stopwords and t not in punc_stop and t not in twitter_stop)]

df['tweet-tokens'] = [nltk.wordpunct_tokenize(tweet) for tweet in df['tweet-text']]
df['tweet-tokens-filtered'] = df['tweet-tokens'].map(remove_stop_punct)

plt.figure(figsize = (10,8))
plt.title("Cookie Monster's Word Frequency")
tweet_tokens_filtered_list = [item for sublist in df["tweet-tokens-filtered"] for item in sublist]
nltk.FreqDist(tweet_tokens_filtered_list).plot(30) #Show me just the top 30


# list for tokenized documents in loop
tweet_texts = []
tweets_set = df['tweet-text']

# preprocessing – clean, tokenize, remove stopwords, and stem
for i in tweets_set:
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [t for t in tokens if (t not in en_stop and t not in twitter_stop)]
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    tweet_texts.append(stemmed_tokens)
 
# build our dictionary and matrix
tweet_dict = gensim.corpora.Dictionary(tweet_texts)
tweet_corpus = [tweet_dict.doc2bow(text) for text in tweet_texts]

tweet_topics = 3
tweet_model = gensim.models.ldamodel.LdaModel(tweet_corpus, num_topics=tweet_topics, id2word = tweet_dict, passes=20)
print(tweet_model.print_topics(tweet_topics, num_words=3))

# assign topic labels to each tweet
topics = []
for i in range(0, len(tweet_texts)):
    topics.append(tweet_model[tweet_dict.doc2bow(tweet_texts[i])])
df["topics"] = topics

df.iloc[0]["tweet-text"]

tweet_vis = pyLDAvis.gensim_models.prepare(tweet_model, tweet_corpus, tweet_dict)
pyLDAvis.display(tweet_vis)
