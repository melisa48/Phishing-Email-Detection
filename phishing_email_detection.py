import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import joblib

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class PhishingDetector:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.vectorizer = None
        self.models = {}
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_data(self):
        self.data = self.data.fillna('')
        self.data['processed_text'] = self.data['text'].apply(self.preprocess_text)
        self.X = self.data['processed_text']
        self.y = np.where(self.data['label'] == 'phishing', 1, 0)

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    def split_data(self, test_size=0.2, random_state=42):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y)

    def create_pipeline(self, classifier):
        return Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', classifier)
        ])

    def train_models(self, X_train, y_train):
        models = {
            'Naive Bayes': self.create_pipeline(MultinomialNB()),
            'SVM': self.create_pipeline(SVC(kernel='linear', probability=True)),
            'Random Forest': self.create_pipeline(RandomForestClassifier())
        }

        for name, model in models.items():
            print(f"Training {name} model...")
            model.fit(X_train, y_train)
            self.models[name] = model

    def evaluate_models(self, X_test, y_test):
        for name, model in self.models.items():
            print(f"\n{name} Model Evaluation:")
            y_pred = model.predict(X_test)
            print("Accuracy:", accuracy_score(y_test, y_pred))
            print("Classification Report:\n", classification_report(y_test, y_pred))
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
            self.plot_roc_curve(model, X_test, y_test, name)

    def plot_roc_curve(self, model, X_test, y_test, model_name):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.show()

    def perform_cross_validation(self, model_name, cv=5):
        model = self.models[model_name]
        scores = cross_val_score(model, self.X, self.y, cv=cv, scoring='accuracy')
        print(f"\n{model_name} Cross-Validation Scores:", scores)
        print(f"Mean CV Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

    def save_model(self, model_name, filename):
        joblib.dump(self.models[model_name], filename)

    def load_model(self, model_name, filename):
        self.models[model_name] = joblib.load(filename)

def main():
    detector = PhishingDetector('emails.csv')
    detector.preprocess_data()

    # Print dataset information
    print(f"Total samples: {len(detector.X)}")
    print(f"Phishing emails: {sum(detector.y)}")
    print(f"Legitimate emails: {len(detector.y) - sum(detector.y)}")

    X_train, X_test, y_train, y_test = detector.split_data()

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    detector.train_models(X_train, y_train)
    detector.evaluate_models(X_test, y_test)

    # Perform cross-validation instead of grid search
    for model_name in detector.models.keys():
        detector.perform_cross_validation(model_name, cv=3)  # Using 3-fold CV due to small dataset

    # Save and load model example
    detector.save_model('SVM', 'svm_model.joblib')
    detector.load_model('SVM', 'svm_model.joblib')

if __name__ == "__main__":
    main()