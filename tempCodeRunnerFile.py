import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Get correct path
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'review.csv')

# Load data
df = pd.read_csv(csv_path)

# Clean column names (important!)
df.columns = df.columns.str.strip()

# Select needed columns
df = df[['body', 'rating']]

# Drop neutral reviews
df = df[df['rating'] != 3]

# Create sentiment labels
df['sentiment'] = df['rating'].apply(lambda x: 'positive' if x >= 4 else 'negative')

# Remove missing text
df = df.dropna(subset=['body'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['body'], df['sentiment'], test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)  # increased iterations for stability
model.fit(X_train_tf, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test_tf))
print("Accuracy:", accuracy)

# Save model properly
with open(os.path.join(script_dir, 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)

with open(os.path.join(script_dir, 'vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model saved.")