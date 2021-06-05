import sys
import pandas as pd
from sqlalchemy import create_engine
pd.options.mode.chained_assignment = None

def load_data(messages_filepath, categories_filepath):
    '''
    This function loads the data given per parameter and merges them into a dataframe
    
    INPUT:
    :param messages_filepath: path to the file messages.csv
    :param categories_filepath: path to the file categories.csv
    
    OUTPUT:
    merged dataframe of the two inputs on the key 'id'
    '''
    
    # loading data into dataframes
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merging data on key 'id'
    df = messages.merge(categories, on='id', how='left')
    
    return df


def clean_data(df):
    '''
    This function cleans the data. The steps are described in the code.
    
    INPUT:
    :param df: dataframe to be cleaned
    
    OUTPUT:
    cleaned dataframe
    '''
    # splitting the categories column by semicolon
    categories = df['categories'].str.split(';', expand=True)
    
    # selecting the frist row of the categories dataframe
    row = categories.iloc[0]

    # splitting the data by '-' and taking the first part of the string
    # this string is then used to rename the column
    # this could also be achieved by row.str.split('-', expand=True)[0]
    category_colnames = row.str.split('-').apply(lambda x:x[0])
    
    # renaming the columns
    categories.columns = category_colnames
    
    # this loop converts the entry of a cell to the last char, which then is converted into an int
    for column in categories:
        # setting each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-').apply(lambda x:x[1])

        # converting column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # dropping the categories column because it's no longer needed
    # could also be achieved by df.drop('categories', inplace=True, axis=1)
    df = df.drop(columns=['categories'])
    
    # concatenating the dataframe df with the categories
    df = pd.concat([df, categories], axis=1)
    
    # The column related has the values [0, 1, 2]
    # In order to use the '2'-value I convert it to a '1'
    # This way it is ensured that the column only contains the binary values [0, 1]
    df.related[df.related == 2] = 1
    
    # dropping duplicate entries
    df = df.drop_duplicates()
    
    return df
    
    

def save_data(df, database_filename):
    '''
    This function saves the dataframe into the database given per parameter
    
    INPUT:
    :param df: dataframe to be saved into the database
    :param database_filename: filename of the database
    
    OUTPUT:
    None
    '''
    
    # creating engine for the desired database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    
    # inserting data into the table 'disaster_response'
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')


def main():
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