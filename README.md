# News Category Classification using N-grams and Bag-of-Words

This project implements a text classification system that categorizes news headlines into different categories using N-grams and Bag-of-Words (BoW) features with a Logistic Regression classifier. The project includes a complete machine learning pipeline and a RESTful API for making predictions.

## Features

- **Text Preprocessing**: Includes cleaning, tokenization, stopword removal, and lemmatization
- **N-gram Features**: Supports unigram, bigram, and trigram features
- **Model Training**: Implements Logistic Regression with hyperparameter tuning
- **Model Evaluation**: Provides detailed classification reports and confusion matrices
- **RESTful API**: FastAPI-based endpoint for making predictions
- **Model Persistence**: Saves the best performing model and label encoder

## Project Structure
text-classification/ ├── data/ # Dataset directory │ └── News_Category_Dataset_v3.json ├── src/ # Source code │ ├── init.py │ ├── train.py # Training script │ ├── api.py # FastAPI application │ └── utils.py # Utility functions ├── models/ # Saved models and label encoder ├── assets/ # Confusion matrix visualizations ├── requirements.txt # Python dependencies └── README.md # This file


## Prerequisites

- Python 3.8+
- pip (Python package installer)

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd text-classification-using-N-grams-and-BOW
Create and activate a virtual environment:
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
bash
pip install -r requirements.txt
Download NLTK data:
python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
Usage
1. Data Preparation
Place your News_Category_Dataset_v3.json file in the data/ directory.

2. Training the Model
Run the training script:

bash
python -m src.train
This will:

Preprocess the data
Train models with different n-gram ranges
Save the best performing model to models/
Generate confusion matrix visualizations in assets/
3. Running the API
Start the FastAPI server:

bash
python -m src.api
The API will be available at http://localhost:8000

4. Making Predictions
You can test the API using curl:

bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text":"The stock market reached a new high today"}'
Or using Python:

python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "The stock market reached a new high today"}
)
print(response.json())
API Endpoints
GET /: API information
POST /predict: Classify a news headline
Request body: {"text": "Your news headline here"}
Response:
json
{
    "text": "Your news headline here",
    "processed_text": "processed version of the text",
    "predicted_category": "BUSINESS",
    "category_id": 3
}
Model Evaluation
The training script evaluates models using:

Accuracy
Precision, Recall, and F1-score per class
Confusion matrix (saved to assets/)
Contributing
Fork the repository
Create a new branch (git checkout -b feature-branch)
Commit your changes (git commit -am 'Add new feature')
Push to the branch (git push origin feature-branch)
Create a new Pull Request
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Dataset: News Category Dataset
NLTK for text processing
scikit-learn for machine learning
FastAPI for the web framework

### Key Features of this README:

1. **Clear Project Overview**: Explains what the project does and its key features
2. **Visual Project Structure**: Shows how the project is organized
3. **Detailed Setup Instructions**: Step-by-step guide to get started
4. **Usage Examples**: Clear commands for training and using the API
5. **API Documentation**: Complete endpoint documentation
6. **Contribution Guidelines**: Instructions for others to contribute
7. **License Information**: Standard MIT license

Would you like me to make any adjustments to this README?
Feedback submitted
Me

create readme with this 

Revert
I'll help you create the README.md file with comprehensive content. Here's the complete content you can use:

markdown
# News Category Classification using N-grams and Bag-of-Words

This project implements a text classification system that categorizes news headlines into different categories using N-grams and Bag-of-Words (BoW) features with a Logistic Regression classifier.

## Features

- **Text Preprocessing**: Includes cleaning, tokenization, stopword removal, and lemmatization
- **N-gram Features**: Supports unigram, bigram, and trigram features
- **Model Training**: Implements Logistic Regression with hyperparameter tuning
- **Model Evaluation**: Provides detailed classification reports and confusion matrices
- **RESTful API**: FastAPI-based endpoint for making predictions
- **Model Persistence**: Saves the best performing model and label encoder

## Project Structure
text-classification/ ├── data/ # Dataset directory │ └── News_Category_Dataset_v3.json ├── src/ # Source code │ ├── init.py │ ├── train.py # Training script │ ├── api.py # FastAPI application │ └── utils.py # Utility functions ├── models/ # Saved models and label encoder ├── assets/ # Confusion matrix visualizations ├── requirements.txt # Python dependencies └── README.md # This file


## Prerequisites

- Python 3.8+
- pip (Python package installer)

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd text-classification-using-N-grams-and-BOW
Create and activate a virtual environment:
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
bash
pip install -r requirements.txt
Download NLTK data:
python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
Usage
1. Data Preparation
Place your News_Category_Dataset_v3.json file in the data/ directory.

2. Training the Model
Run the training script:

bash
python -m src.train
This will:

Preprocess the data
Train models with different n-gram ranges
Save the best performing model to models/
Generate confusion matrix visualizations in assets/
3. Running the API
Start the FastAPI server:

bash
python -m src.api
The API will be available at http://localhost:8000

4. Making Predictions
You can test the API using curl:

bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text":"The stock market reached a new high today"}'
API Endpoints
GET /: API information
POST /predict: Classify a news headline
Request body: {"text": "Your news headline here"}
Response:
json
{
    "text": "Your news headline here",
    "processed_text": "processed version of the text",
    "predicted_category": "BUSINESS",
    "category_id": 3
}
Contributing
Fork the repository
Create a new branch (git checkout -b feature-branch)
Commit your changes (git commit -am 'Add new feature')
Push to the branch (git push origin feature-branch)
Create a new Pull Request
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Dataset: News Category Dataset
NLTK for text processing
scikit-learn for machine learning
FastAPI for the web framework
