import os
import re
import nltk
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_text(text):
    """Preprocess text data."""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars and numbers
    
    # Tokenization and stopword removal
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if len(word) > 1]
    
    return ' '.join(tokens)

def plot_confusion_matrix(y_true, y_pred, classes, model_name, save_dir='assets', normalize=False):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = f'Normalized Confusion Matrix - {model_name}'
    else:
        title = f'Confusion Matrix - {model_name}'
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 8})
    
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot
    filename = f"{save_dir}/confusion_matrix_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
    if normalize:
        filename += "_normalized"
    filename += ".png"
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {filename}")

def save_model(model, label_encoder, model_name, model_dir='models'):
    """Save the model and label encoder to disk."""
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_path = f'{model_dir}/{model_name}.joblib'
    joblib.dump(model, model_path)
    
    # Save the label encoder
    encoder_path = f'{model_dir}/label_encoder.joblib'
    joblib.dump(label_encoder, encoder_path)
    
    print(f"Model saved to {model_path}")
    print(f"Label encoder saved to {encoder_path}")

def load_model(model_dir='models'):
    """Load the saved model and label encoder."""
    model = joblib.load(f'{model_dir}/best_model.joblib')
    label_encoder = joblib.load(f'{model_dir}/label_encoder.joblib')
    return model, label_encoder