import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle


df = pd.read_csv('news.csv')  


X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)


vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)


model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)


pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved successfully.")





















