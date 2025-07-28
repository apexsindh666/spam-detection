import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import joblib

# 1. Load dataset
df = pd.read_csv("data/spam.csv", encoding='latin1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df.dropna(inplace=True)

# 2. Balance the data
ham = df[df.label == 0]
spam = df[df.label == 1]
spam_upsampled = resample(spam, replace=True, n_samples=len(ham), random_state=42)
df_balanced = pd.concat([ham, spam_upsampled])

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced['message'], df_balanced['label'], test_size=0.2, random_state=42)

# 4. TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)

# 5. Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 6. Save model and vectorizer
joblib.dump(model, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("✅ model.joblib and vectorizer.joblib saved.")
