# spam-detection
📌 Project Title: SMS Spam Detection using Logistic Regression and TF-IDF (NLP)

📍 Overview:
This project is an end-to-end Natural Language Processing (NLP) solution aimed at detecting whether a given SMS message is spam or ham (not spam). The solution is built using Python, with Scikit-learn for machine learning, Pandas for data handling, and Matplotlib/Seaborn for visualization. This model helps filter out spam messages with high accuracy and provides clear visual insights into which words are most indicative of spam.

📌 Problem Statement:
Spam messages are a common problem in digital communication. The objective of this project is to create a binary classification model that can automatically detect spam messages using machine learning techniques and understand which terms are most associated with spam.

📍 Data Source:
The dataset used is the classic SMS Spam Collection Dataset, which contains 5,572 labeled SMS messages, categorized into "ham" (legitimate) or "spam" (unwanted). The dataset was preprocessed and loaded using Pandas, with appropriate handling for missing values and renaming of relevant columns.

📍 Approach:

Data Preprocessing:
Removed unnecessary columns and null values.
Converted categorical labels ("ham" and "spam") into binary format (0 and 1).
Balanced the dataset by upsampling the minority class (spam) using Scikit-learn's resample to ensure equal representation of both classes in the training data.
Feature Extraction:
Used TF-IDF Vectorizer with bi-grams (ngram_range = (1, 2)) and English stop words removed. This helped extract meaningful textual features while reducing noise.
Model Building:
Applied Logistic Regression, a lightweight yet powerful linear model, to classify messages.
The model was trained on the balanced dataset and evaluated using a standard 80-20 train-test split.
Evaluation Metrics:
Achieved 99.2% accuracy on the test data.
Used classification metrics like precision, recall, and F1-score, all of which were above 0.99 for both classes.
Results show strong generalization and minimal overfitting.
📍 Message Prediction Function:
A custom Python function predict_message() was implemented to classify any new message as "Spam" or "Ham" using the trained model and vectorizer pipeline. This function makes the system user-friendly and easily extendable to real-time applications.

📍 Visualization:

Extracted the top 20 most influential words in spam messages based on TF-IDF scores.
Used Seaborn to create an aesthetically pleasing bar plot showing these top spam words, offering insights into common patterns used in spam messages (e.g., "free", "win", "claim").
📍 Tools & Libraries Used:

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, WordCloud
📍 Outcome:
This project provided hands-on experience in handling imbalanced datasets, performing text vectorization using TF-IDF, and building an interpretable and accurate binary classification model. The visualization component added explainability to the model’s predictions, which is crucial in real-world applications.

📍 Future Scope:

Deploy the model using Flask or Streamlit.
Integrate with messaging platforms for real-time spam detection.
Experiment with advanced models like Naive Bayes, SVM, or deep learning (e.g., LSTM) for further performance improvements.
