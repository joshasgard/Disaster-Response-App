import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Function to extract raw .csv data files containing messages and 
       their categories, and then merge them. 
    
    Args: 
        messages_filepath(str): path to raw .csv message file.
        categories_filepath(str): path to raw .csv message categories file.

    Returns:
        dataframe: containing messages and respective categories ready for cleaning.
    
    """
    #   Import data files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    #   Merge datasets
    df = pd.merge(messages, categories)
    
    return df


def clean_data(df):
    """Function to transform dataset for use in an ML Pipeline.
       Splits the categories into separate category columns, cleans class values
       for each category, removes duplicates. 
       
    Args:
        df: dataframe to be cleaned.

    Returns:
        dataframe(df): ML-ready dataset.
 
    """
    #   Split the categories column into separate columns 
    categories_table = df.categories.str.split(';',expand = True)
    categories_table.columns = categories_table.iloc[0].apply(lambda x: x[0:-2]) 
    
    #   Clean class values for each category and cast as numeric
    for column in categories_table:
        categories_table[column] = categories_table[column].str[-1].astype('int')

    #   Replace 'categories' column in df with new columns
    df.drop('categories', axis = 1, inplace=True)
    df = pd.concat([df,categories_table], axis=1)

    #   Remove dupplicates from data
    df = df.drop_duplicates(subset='message')

    return df


def save_data(df, database_filename):
    """Function to create SQL database engine and to load transformed data into 
       the database. 
    
    Args: 
        dataframe(df): transformed dataset.
        database_filename(str): SQL database engine name. 

    Returns:
        None
    
    """
    
    #   Create SQL engine with specified filename
    engine = create_engine(database_filename)

    #   Load cleaned data into SQL engine, replacing data in database if defined 
    #   name already exists.
    df.to_sql('cleaned_messages', engine, index = False, if_exists = 'replace')
      


def main():
    """
        Function to run the whole script and manages user arguments specifying files and 
        database engine name.

        Args: 
            None
        
        Returns:
            None
       
    """

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()