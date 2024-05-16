import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch



#for bert
"""We are using pretrained 'bert-base-multilingual-uncased-sentiment' model 
for predicting the sentiment of the review as a number of stars (between 1 and 5)
""";
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1






# Vader sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # for polarity score
analyser = SentimentIntensityAnalyzer()
"""following functions returns positive, negative, neutral emotion score of the text respectively. 
""";
def pos(tweet):
    SentDict = analyser.polarity_scores(tweet)
    return SentDict['pos']
def neg(tweet):
    SentDict = analyser.polarity_scores(tweet)
    return SentDict['neg']
def neu(tweet):
    SentDict = analyser.polarity_scores(tweet)
    return SentDict['neu']

df=pd.read_csv('data/IndianElection19TwitterData.csv',index_col=0)





# Tweets related to Narendra Modi
"""Filtering out tweets with some keywords and hashtags in it
   referring to Narendra Modi that are commonly used on twitter
""";
modi = ["Modi","PM","modi", "#PMModi","modi ji", "narendra modi", "@narendramodi","#Vote4Modi"]
modi_df = pd.DataFrame(columns=["Date", "User","Tweet"])
def ismodi(tweet):
    t = tweet.split()
    for i in modi:
        if i in t:
            return True
modi_rows = []
# Here df is the main data
for row in df.values:
    if ismodi(str(row[2])):
         modi_rows.append({"Date":row[0], "User":row[1],"Tweet":row[2]})
modi_df = pd.concat([modi_df, pd.DataFrame(modi_rows)])
# print(modi_df.head(10))
modi_df['Tweet'].nunique()



# Tweets related to Rahul Gandhi
"""
 Filtering out tweets with some keywords and hashtags in it 
 referring to Rahul Gandhi that are commonly used on twitter
""";
rahul = ["rahul", "Rahul","RahulGandhi", "gandhi","@RahulGandhi","Gandhi","#Vote4Rahul","#Vote4Gandhi","#Vote4RahulGandhi"]
rahul_df = pd.DataFrame(columns=["Date", "User","Tweet"])
def israhul(tweet):
    t = tweet.split()
    for i in rahul:
        if i in t:
            return True
rahul_rows = []
for row in df.values:
    if israhul(str(row[2])):
         rahul_rows.append({"Date":row[0], "User":row[1],"Tweet":row[2]})
rahul_df = pd.concat([rahul_df, pd.DataFrame(rahul_rows)])



# Data Cleaning : Removing Stopwords & Panctuations
from sklearn.feature_extraction import text
import string
stop = text.ENGLISH_STOP_WORDS
"""Removing stopwords (as in sklearn library) from tweets so as to get good polarity scores
""";
modi_df['Tweet'] = modi_df['Tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
rahul_df['Tweet'] = rahul_df['Tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text
"""Removing panctuations from tweets
""";
modi_df['Tweet'] = modi_df['Tweet'].apply(remove_punctuations)
rahul_df['Tweet'] = rahul_df['Tweet'].apply(remove_punctuations)



# VadarSentiment Sentiment Analysis
"""Calculating the polarity scores with help of code snippets mentioned at the importing libraries section
""";
modi_df['pos'] = modi_df['Tweet'].apply(lambda x :pos(x))
modi_df['neg'] = modi_df['Tweet'].apply(lambda x :neg(x))
modi_df['neu'] = modi_df['Tweet'].apply(lambda x :neu(x))
emotion=[]
for i in range(0,25683):
    emotion.append(max(modi_df['pos'][i],modi_df['neu'][i],modi_df['neg'][i]))
modi_df['FinalEmotion']=emotion

"""Traversing through the polarity scores for each tweet and
assigning the Final Emotion as per the highest score among positive, negative, neutral
""";
for i in range(0,25683):
    if modi_df['FinalEmotion'][i]==modi_df['pos'][i]:
        modi_df['FinalEmotion'][i]='positive'
    elif modi_df['FinalEmotion'][i]==modi_df['neg'][i]:
        modi_df['FinalEmotion'][i]='negative'
    elif modi_df['FinalEmotion'][i]==modi_df['neu'][i]:
        modi_df['FinalEmotion'][i]='neutral'
# print(modi_df.head(20));
print(modi_df['FinalEmotion'].value_counts());

#TODO: FOR RAHUL
"""
 Calculating the polarity scores with help of code snippets mentioned at the importing libraries section
    """;
rahul_df['pos'] = rahul_df['Tweet'].apply(lambda x :pos(x))
rahul_df['neg'] = rahul_df['Tweet'].apply(lambda x :neg(x))
rahul_df['neu'] = rahul_df['Tweet'].apply(lambda x :neu(x))

emotion=[]
for i in range(0,14148):
    emotion.append(max(rahul_df['pos'][i],rahul_df['neu'][i],rahul_df['neg'][i]))

rahul_df['FinalEmotion']=emotion
print(rahul_df['FinalEmotion'].value_counts());

"""
 Traversing through the polarity scores for each tweet and
 assigning the Final Emotion as per the highest score among positive, negative, neutral
    """;
for i in range(0,14148):
    if rahul_df['FinalEmotion'][i]==rahul_df['pos'][i]:
        rahul_df['FinalEmotion'][i]='positive'
    elif rahul_df['FinalEmotion'][i]==rahul_df['neg'][i]:
        rahul_df['FinalEmotion'][i]='negative'
    elif rahul_df['FinalEmotion'][i]==rahul_df['neu'][i]:
        rahul_df['FinalEmotion'][i]='neutral'

rahul_df['FinalEmotion'].value_counts()



# Sentiments for Narendra Modi
plt.figure(figsize=(15,10))
sns.set_style("darkgrid")
ax = sns.countplot(x=modi_df['FinalEmotion'],palette=['#36454F','#89CFF0'])
ax.set_title('Sentiments scores of Tweets about Modi')
plt.savefig('data/sentiment_plot1.png', dpi=300)

# Sentiments for Rahul
plt.figure(figsize=(15,10))
sns.set_style("darkgrid")
ax = sns.countplot(x=rahul_df['FinalEmotion'],palette=['#36454F','#89CFF0'])
ax.set_title('Sentiments scores of Tweets about Rahul')
plt.savefig('data/sentiment_plot2.png', dpi=300)




#Just to clear the previous sentiments by vaderSentiment, we need to drop that columns for using Flair on it
rahul_df.drop(['pos', 'neg', 'neu', 'FinalEmotion'],axis=1,inplace=True)
modi_df.drop(['pos', 'neg', 'neu', 'FinalEmotion'],axis=1,inplace=True)



#TextBlob Sentiment Analysis
from textblob import TextBlob
def textblob_prediction(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "pos"
    elif polarity < 0:
        return "neg"
    else:
        return "neu"

# Apply sentiment analysis using TextBlob
rahul_df['Emotion'] = rahul_df['Tweet'].apply(textblob_prediction)
modi_df['Emotion'] = modi_df['Tweet'].apply(textblob_prediction)
print(modi_df['Emotion'].value_counts());
print(rahul_df['Emotion'].value_counts());

# Sentiments for Narendra Modi
plt.figure(figsize=(15,10))
sns.set_style("darkgrid")
ax = sns.countplot(x=modi_df['Emotion'],palette=['#36454F','#89CFF0'])
ax.set_title('Sentiments scores of Tweets about Modi')
plt.savefig('data/sentiment_plot3.png', dpi=300)

# Sentiments for Rahul
plt.figure(figsize=(15,10))
sns.set_style("darkgrid")
ax = sns.countplot(x=rahul_df['Emotion'],palette=['#36454F','#89CFF0'])
ax.set_title('Sentiments scores of Tweets about Rahul')
plt.savefig('data/sentiment_plot4.png', dpi=300)




modi_df.drop(['Emotion'],axis=1,inplace=True)
rahul_df.drop(['Emotion'],axis=1,inplace=True)




from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
svr_regressor, tfidf_vectorizer = joblib.load('svr_model_with_tfidf.pkl')

def svr_prediction(text):
    text_features = tfidf_vectorizer.transform([text])
    sentiment_score = svr_regressor.predict(text_features)[0]
    if sentiment_score > 0:
        return "pos"
    elif sentiment_score < 0:
        return "neg"
    else:
        return "neu"
#
rahul_df['Emotion'] = rahul_df['Tweet'].apply(svr_prediction)
modi_df['Emotion'] = modi_df['Tweet'].apply(svr_prediction)
print(modi_df['Emotion'].value_counts());
print(rahul_df['Emotion'].value_counts());


# Print a few samples with their predicted sentiments
print("Samples with Predicted Sentiments:")
for index, row in rahul_df.sample(5).iterrows():
    print("Tweet:", row['Tweet'])
    print("Predicted Emotion:", row['Emotion'])
    print()



# Sentiments for Narendra Modi
plt.figure(figsize=(15,10))
sns.set_style("darkgrid")
ax = sns.countplot(x=modi_df['Emotion'],palette=['#36454F','#89CFF0'])
ax.set_title('Sentiments scores of Tweets about Modi')
plt.savefig('data/sentiment_plot5.png', dpi=300)

# Sentiments for Rahul
plt.figure(figsize=(15,10))
sns.set_style("darkgrid")
ax = sns.countplot(x=rahul_df['Emotion'],palette=['#36454F','#89CFF0'])
ax.set_title('Sentiments scores of Tweets about Rahul')
plt.savefig('data/sentiment_plot6.png', dpi=300)























