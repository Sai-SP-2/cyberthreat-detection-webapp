import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Basic preprocessing for event vector generation
def preprocess_data(df):
    possible_targets = ['class', 'label', 'target', 'Category', 'labels']
    target_col = next((col for col in df.columns if col in possible_targets), None)
    if not target_col:
        raise ValueError(f"Target column not found. Available columns: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical columns
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    # Encode target
    y_encoded = LabelEncoder().fit_transform(y)
    num_classes = len(np.unique(y_encoded))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    return {
        'ml': (X_train, X_test, y_train, y_test),
        'num_classes': num_classes
    }

# TF-IDF preprocessing
def tfidf_preprocess(df):
    text_col = next((col for col in df.columns if df[col].dtype == 'object'), None)
    if not text_col:
        raise ValueError("No suitable text column found for TF-IDF.")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df[text_col])
    return tfidf_matrix

# Generate vector for ML models
def event_vector_generation(df):
    data = preprocess_data(df)
    X_train, X_test, y_train, y_test = data['ml']
    num_classes = data['num_classes']
    return X_train, X_test, y_train, y_test, num_classes


# Evaluation function
def evaluate_model(model, X_test, y_test):
    prediction_prob = model.predict(X_test)
    prediction_data = np.argmax(prediction_prob, axis=1) if prediction_prob.ndim > 1 else prediction_prob
    y_true = np.asarray(y_test).flatten()

    accuracy = accuracy_score(y_true, prediction_data) * 100
    precision = precision_score(y_true, prediction_data, average='macro') * 100
    recall = recall_score(y_true, prediction_data, average='macro') * 100
    f1 = f1_score(y_true, prediction_data, average='macro') * 100

    return accuracy, precision, recall, f1

# ANN model
def run_ann(X_train, y_train, X_test, y_test, num_classes):
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, verbose=0)
    return evaluate_model(model, X_test, y_test)

# Other classic ML models
def run_knn(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test)

def run_svm(X_train, y_train, X_test, y_test):
    model = SVC()
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test)

def run_dt(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test)

def run_rf(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test)

def run_nb(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test)
