import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix      
from sklearn.utils import resample
from .utils import preprocess_text, plot_confusion_matrix, save_model

def train_and_evaluate(X_train, X_test, y_train, y_test, ngram_range, model_name, label_encoder):
    """Train and evaluate a model with given n-gram range."""
    print(f"\nTraining {model_name} (ngram_range={ngram_range})...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=ngram_range)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print classification report
    print(f"\n{model_name} Results:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Accuracy: {accuracy:.4f}")
    
    # Plot confusion matrices
    class_names = label_encoder.classes_
    plot_confusion_matrix(y_test, y_pred, class_names, model_name)
    plot_confusion_matrix(y_test, y_pred, class_names, model_name, normalize=True)
    
    return pipeline, accuracy

def main():
    # 1. Load and prepare data
    print("Loading dataset...")
    df = pd.read_json("data/News_Category_Dataset_v3.json", lines=True)
    df = df[["headline", "category"]]
    
    # 2. Balance the dataset
    print("\nBalancing dataset...")
    max_samples = df['category'].value_counts().max()
    df_balanced = pd.DataFrame()
    
    for category in df['category'].unique():
        category_samples = df[df['category'] == category]
        n_samples = max_samples - len(category_samples)
        
        if n_samples > 0:
            upsampled = resample(category_samples,
                               replace=True,
                               n_samples=n_samples,
                               random_state=42)
            df_balanced = pd.concat([df_balanced, category_samples, upsampled])
        else:
            df_balanced = pd.concat([df_balanced, category_samples])
    
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 3. Preprocess text
    print("\nPreprocessing text...")
    df_balanced['processed_text'] = df_balanced['headline'].apply(preprocess_text)
    
    # 4. Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    df_balanced['category_encoded'] = label_encoder.fit_transform(df_balanced['category'])
    
    # 5. Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df_balanced['processed_text'],
        df_balanced['category_encoded'],
        test_size=0.2,
        random_state=42,
        stratify=df_balanced['category_encoded']
    )
    
    # 6. Train and evaluate different n-gram models
    models = [
        ((1, 1), "Unigram_BOW"),
        ((1, 2), "Bigram"),
        ((1, 3), "Trigram")
    ]
    
    best_accuracy = 0
    best_model = None
    best_model_name = ""
    
    for ngram_range, name in models:
        model, accuracy = train_and_evaluate(X_train, X_test, y_train, y_test, 
                                           ngram_range, name, label_encoder)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
    
    # Save the best model and label encoder
    print(f"\nSaving the best model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    save_model(best_model, label_encoder, "best_model")
    
    return best_model, label_encoder

if __name__ == "__main__":
    best_model, label_encoder = main()
