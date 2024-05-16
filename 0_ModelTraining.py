import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import joblib

# Read the cleaned CSV file into a pandas DataFrame
df = pd.read_csv("data/CleanedDataset.csv")

# Preprocess the text data (if further preprocessing is needed)

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Convert text data into numerical vectors
X = tfidf_vectorizer.fit_transform(df['full_text'])
y = df['Compound']  # Assuming 'Compound' is the sentiment score

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Support Vector Regression (SVR) model
svr_regressor = SVR(kernel='linear')
svr_regressor.fit(X_train, y_train)

# Evaluate the model
y_pred = svr_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

joblib.dump((svr_regressor, tfidf_vectorizer), 'svr_model_with_tfidf.pkl')
