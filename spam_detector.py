import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load and prepare data
def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path, encoding='latin1')
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    df = df[['label', 'message']]

    df.dropna(inplace=True)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Balance the dataset
    ham = df[df.label == 0]
    spam = df[df.label == 1]
    spam_upsampled = resample(spam, replace=True, n_samples=len(ham), random_state=42)
    df_balanced = pd.concat([ham, spam_upsampled])

    return df_balanced

# Train and save model
def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Save model and vectorizer
    joblib.dump(model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    print("\n💾 Model and vectorizer saved successfully!")

# Predict new message
def predict_message(message):
    model = joblib.load('model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    vec = vectorizer.transform([message])
    pred = model.predict(vec)[0]
    return "Spam" if pred == 1 else "Ham"

# Run this part once to train and save the model
if __name__ == "__main__":
    df = load_and_prepare_data("data/spam.csv")
    train_model(df)
