import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.feature_extraction import text
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


# st.set_option('deprecation.showPyplotGlobalUse', False)

svr_regressor, tfidf_vectorizer = joblib.load('data/svr_model_with_tfidf.pkl')

def load_data(csv_file):
    return pd.read_csv(csv_file)

def vader_sentiment(tweet):
    analyser = SentimentIntensityAnalyzer()
    SentDict = analyser.polarity_scores(tweet)
    if SentDict['compound'] >= 0.05:
        return "pos"
    elif SentDict['compound'] <= -0.05:
        return "neg"
    else:
        return "neu"

def textblob_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "pos"
    elif polarity < 0:
        return "neg"
    else:
        return "neu"

def svr_sentiment(text):
    text_features = tfidf_vectorizer.transform([text])
    sentiment_score = svr_regressor.predict(text_features)[0]
    if sentiment_score > 0:
        return "pos"
    elif sentiment_score < 0:
        return "neg"
    else:
        return "neu"

def preprocess_tweet(tweet):
    stop = text.ENGLISH_STOP_WORDS
    tweet = ' '.join([word for word in tweet.split() if word not in stop])
    for punctuation in string.punctuation:
        tweet = tweet.replace(punctuation, '')
    return tweet

def plot_sentiment(df, title):
    plt.figure(figsize=(7.5, 5))
    sns.set(style="whitegrid")

    ax = sns.countplot(x=df['Emotion'], palette=['#36454F', '#89CFF0', '#FFD700'])
    ax.set_title(title, fontsize=20, fontweight='bold', fontname='Helvetica')
    ax.set_xlabel('Sentiment', fontsize=14, fontname='Helvetica')
    ax.set_ylabel('Count', fontsize=14, fontname='Helvetica')

    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    st.pyplot()


def display_tweets(df):
    total_tweets = len(df)
    st.sidebar.markdown(
        f"""
        <style>
        .tweet-container {{
            padding: 20px;
            border: 1px solid #ccc;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            background-color: #f9f9f9;
            text-align: center;  /* Center the tweets */
        }}
        .tweet {{
            margin-bottom: 20px;  /* Add space between tweets */
            margin-top: 20px; /* Add top margin to tweets */
        }}
        .twitter-logo {{
            width: 50px;
            margin-bottom: 10px;
        }}
        .tweet-info {{
            font-size: 16px;
            margin-top: 5px;
        }}
        </style>
        <div class='tweet-container'>
            <h2 class='twitter-text'>Twitter</h2>
            <div class='tweet-info'>Total Tweets: {total_tweets}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    tweet_container = st.sidebar.empty()
    while True:
        for index, tweet in df.iterrows():
            with tweet_container.container():
                st.markdown(f"<div class='tweet' style='background-image: url(twitter_logo.jpg); background-size: contain; background-repeat: no-repeat;'>Tweet_{index}: {tweet['text']}</div>", unsafe_allow_html=True)
                time.sleep(2)


def main():
    st.title("Indian Election Twitter Sentiment Analysis")

    uploaded_file = st.file_uploader("Upload the main CSV file:", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df_copy = df.copy()

        st.write("Data Loaded Successfully!")
        st.write("Generating sentiment analysis plots...")


        # Preprocess tweets
        df['text'] = df['text'].apply(preprocess_tweet)

        # Separate dataframes for Rahul Gandhi and Narendra Modi
        rahul_keywords = ["rahul", "Rahul", "RahulGandhi", "gandhi", "@RahulGandhi", "Gandhi", "#Vote4Rahul", "#Vote4Gandhi", "#Vote4RahulGandhi"]
        modi_keywords = ["Modi", "PM", "modi", "#PMModi", "modi ji", "narendra modi", "@narendramodi", "#Vote4Modi"]
        rahul_df = df[df['text'].str.contains('|'.join(rahul_keywords), case=False)]
        modi_df = df[df['text'].str.contains('|'.join(modi_keywords), case=False)]

        # Sentiment analysis
        rahul_df['Emotion'] = rahul_df['text'].apply(lambda x: max(vader_sentiment(x), textblob_sentiment(x), svr_sentiment(x)))
        modi_df['Emotion'] = modi_df['text'].apply(lambda x: max(vader_sentiment(x), textblob_sentiment(x), svr_sentiment(x)))

        # Plot sentiment analysis for Rahul Gandhi and Narendra Modi
        plot_sentiment(rahul_df, "Sentiment Scores of Tweets about Rahul Gandhi")
        # display_tweets(rahul_df)

        plot_sentiment(modi_df, "Sentiment Scores of Tweets about Narendra Modi")

        display_tweets(df_copy)

if __name__ == "__main__":
    main()
