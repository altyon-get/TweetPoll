import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import joblib

df = pd.read_csv("data/CleanedDataset.csv", nrows=10000)  

vectorizers = {
    "TF-IDF": TfidfVectorizer(),
    "CountVectorizer": CountVectorizer(),
    "HashingVectorizer": HashingVectorizer()
}

svr_regressor = SVR(kernel='linear')

mse_results = {}

for vec_name, vectorizer in vectorizers.items():
    X = vectorizer.fit_transform(df['full_text'])
    y = df['Compound']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svr_regressor.fit(X_train, y_train)

    y_pred = svr_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_results[vec_name] = mse
    print(f"MSE for {vec_name}: {mse}")

best_vectorizer_name = min(mse_results, key=mse_results.get)
best_vectorizer = vectorizers[best_vectorizer_name]
# joblib.dump((svr_regressor, best_vectorizer), f'best_model_with_{best_vectorizer_name}.pkl')
