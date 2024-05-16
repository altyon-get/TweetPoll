import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import joblib

# Read the cleaned CSV file into a pandas DataFrame
df = pd.read_csv("data/CleanedDataset.csv", nrows=10000)  # Read only 10K rows

# Preprocess the text data (if further preprocessing is needed)

# Define a list of vectorizers
vectorizers = {
    "TF-IDF": TfidfVectorizer(),
    "CountVectorizer": CountVectorizer(),
    "HashingVectorizer": HashingVectorizer()
}

# Initialize SVR regressor
svr_regressor = SVR(kernel='linear')

# Dictionary to store MSE for each vectorizer
mse_results = {}

# Iterate over each vectorizer
for vec_name, vectorizer in vectorizers.items():
    # Convert text data into numerical vectors
    X = vectorizer.fit_transform(df['full_text'])
    y = df['Compound']  # Assuming 'Compound' is the sentiment score

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train SVR model
    svr_regressor.fit(X_train, y_train)

    # Evaluate the model
    y_pred = svr_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_results[vec_name] = mse
    print(f"MSE for {vec_name}: {mse}")

# Save the model with the best performance
best_vectorizer_name = min(mse_results, key=mse_results.get)
best_vectorizer = vectorizers[best_vectorizer_name]
# joblib.dump((svr_regressor, best_vectorizer), f'best_model_with_{best_vectorizer_name}.pkl')
