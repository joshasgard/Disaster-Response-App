#   Import libraries and downloads
import sys
import numpy as np
import pandas as pd

#   Import natural language processing toolkit downloads
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#   Import database engine calling form
from sqlalchemy import create_engine

#   Import necessary scikit-learn machine learning pipeline tools
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score

#   Import joblib to save model
import joblib

def load_data(database_filepath):
    """Function to load cleaned data from app database as dataframe.
    
    Args: 
        database_filepath (str): path to database to load from.
    Returns:
        X (series):             messages to classify.
        y (dataframe):          dataframe containing message categories.
        category_names (list):  names of the categories.

    """
    #   load data from database
    engine = create_engine(database_filepath) # calls the database engine
    df = pd.read_sql_table('clean_data', engine) 
    
    #   specify messages and classification categories
    X = df['message']
    y = df.drop(['id','message', 'original','genre'], axis =1)

    #   save category names
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
    """Function converts raw messages into tokens, cleans the tokens and removes
        stopwords.
    
    Args:
        text(str): raw message data to be classified.
    Returns:
        clean_tokens(list): cleaned list of tokens(words).
        
    """
    #   convert each text input into tokens
    tokens = word_tokenize(text)

    #   initialize lemmatizer for converting tokens to root
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    #   remove stopwords    
    clean_tokens = [x for x in clean_tokens if x not in stopwords.words('english')]

    return clean_tokens

def build_model():
    """Builds the machine learning pipeline containing transformers and 
        a final MultiOutput estimator.
        
    Args: 
        None.
    Returns:
        pipeline: defined machine learning pipeline.
    
    """

    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tf_idf', TfidfTransformer()),
                    ('multi_clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=20)))]                   
                        )
    
    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    """Evaluates the performance of trained model.
    
    Args:
        model:          trained model.
        X_test:         test data for prediction.
        y_test:         test classification data for evaluating model predictions.
        category_names: category names.
        
    Returns:
        prints out metric scores - Precision, Recall and Accuracy.
        
    """
    #   predict classes for X_test
    prediction = model.predict(X_test)

    #   loop across each category to print out model metrics
    for i,label in  zip(range (36),category_names):
        precision = round(precision_score(np.array(y_test)[:,i], prediction[:,i], labels = np.unique(prediction[:,i]), average='weighted'), 3)
        recall = round(recall_score(np.array(y_test)[:,i], prediction[:,i], labels = np.unique(prediction[:,i]), average='weighted'), 3)
        accuracy = round(accuracy_score(np.array(y_test)[:,i], prediction[:,i]),3)
        print(i+1,'-',label.upper())
        print('Precision: {}, Recall {}, Accuracy: {}'.format(precision, recall, accuracy))
        print('\n')


def save_model(model, model_filepath):
    """Function to save trained model as a pickle file. 
    
    Args: 
        model: trained model.
        model_filepath: preferred name for saving model.

    Returns:
        None
    
    """
    joblib.dump(model, model_filepath)


def main():
    """ Run the script and handle user arguments.
       
    Args: 
        None
        
    Returns:
        None
       
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()