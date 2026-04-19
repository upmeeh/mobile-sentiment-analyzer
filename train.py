import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'review.csv')

# Load data
df = pd.read_csv(csv_path)

# Clean column names (avoids hidden space bugs)
df.columns = df.columns.str.strip()

# Keep required columns
df = df[['body', 'rating']]

# Drop missing values
df = df.dropna(subset=['body', 'rating'])

# ----- 3-CLASS MAPPING -----
def map_sentiment(r):
    if r >= 4:
        return 'positive'
    elif r == 3:
        return 'neutral'
    else:
        return 'negative'

df['sentiment'] = df['rating'].apply(map_sentiment)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['body'], df['sentiment'], test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tf, y_train)

# Evaluation
y_pred = model.predict(X_test_tf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Report:\n", classification_report(y_test, y_pred))

# Save model
with open(os.path.join(script_dir, 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)

with open(os.path.join(script_dir, 'vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model saved.")