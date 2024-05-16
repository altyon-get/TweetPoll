import joblib

# Load the trained SVR model and TF-IDF vectorizer
svr_model, tfidf_vectorizer = joblib.load('svr_model_with_tfidf.pkl')

# Define some political tweets related to India
political_tweets = [
    "The government's new policy on education is a step in the right direction. #IndiaEducation",
    "I am deeply concerned about the rising unemployment rates in our country. #IndiaUnemployment",
    "The recent budget announcement has neglected crucial sectors like healthcare. #IndiaBudget",
    "It's time for politicians to prioritize environmental conservation over economic gains. #IndiaEnvironment",
    "The infrastructure development projects are transforming the face of rural India, Greak Work. #IndiaInfrastructure #GreatWork"
]

# Predict sentiment for each political tweet
for tweet in political_tweets:
    # Preprocess the tweet
    tweet_vector = tfidf_vectorizer.transform([tweet])
    # Predict sentiment using the SVR model
    sentiment_score = svr_model.predict(tweet_vector)[0]
    print(f"Tweet: {tweet}\nSentiment Score: {sentiment_score}\n")
