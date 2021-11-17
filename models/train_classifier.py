#   Import libraries and downloads
import sys
import os
import numpy as np
import pandas as pd
import time


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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

#   Import joblib to save model
import pickle

#   Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def load_data(database_filename):
    """Function to load cleaned data from app database as dataframe.
    
    Args: 
        database_table (str): saved database table
    Returns:
        X (series):             messages to classify.
        y (dataframe):          dataframe containing message categories.
        category_names (list):  names of the categories.

    """
    #   load data from database
    #   Create SQL engine with database name
    engine = create_engine('sqlite:///'+database_filename)

    #   extract table name from database name
    table_name = os.path.basename(database_filename).split('.')[0]
    
    df = pd.read_sql_table(table_name, engine) 
    
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
                    ('multi_clf', MultiOutputClassifier(RandomForestClassifier()))]                   
                        )

    #   Pipeline Hyperparamenter tuning - remove '#' to include other parameters as you like. 
    #   Training could take several minutes or hours depending on your device and hyperparameter choice
    parameters = {
        'vect__ngram_range': ((1,1), (1,2)),
        #'vect__max_features': (None, 5000, 10000),
        #'tf_idf__use_idf': (True, False),
        #'multi_clf__estimator__min_samples_leaf':[1,2],
        #'multi_clf__estimator__n_estimators': [10,20,100],
        #'multi_clf__estimator__max_depth': [None,5,10],
        #'multi_clf__estimator__min_samples_split': [2,3,5]
        }

    optimizer = GridSearchCV(pipeline, param_grid=parameters)

    return optimizer


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

    #   print out model precision, recall and accuracy
    print(classification_report(y_test, prediction, target_names=category_names))


def save_model(model, model_filepath):
    """Function to save trained model as a pickle file. 
    
    Args: 
        model: trained model.
        model_filepath: preferred name for saving model.

    Returns:
        None
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ Run the script and handle user arguments.
       
    Args: 
        None
        
    Returns:
        None
       
    """
    if len(sys.argv) == 3:
        database_filename, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filename))
        X, y, category_names = load_data(database_filename)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()

        #   training start time
        start_time = time.time()

        print('Training model...')
        model.fit(X_train, y_train)
        
        # training time taken
        print("...Training Time: %s seconds ---" % (time.time() - start_time))

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the name of table of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py DisasterResponse classifier.pkl')


if __name__ == '__main__':
    main()