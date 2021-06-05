import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import joblib


stopWords = stopwords.words('english') # loading the stopwords


# idea was taken from the StartingVerbExtractor from the udacity course
class TextLengthExtractor(BaseEstimator, TransformerMixin):
    '''
    An estimator that can count the text length of each cell in the X
    
    '''
    def fit(self, X, y=None):
        '''
        Return self
        '''
        return self

    def transform(self, X):
        '''
        Count the text length of each cell in the X
        '''
        X_length = pd.Series(X).str.len()
        return pd.DataFrame(X_length)


def load_data(database_filepath):
    '''
    This method loads the data from the database given per parameter
    The data is taken from the table 'disaster_response'
    
    INPUT:
    :param database_filepath: filename of the database
    
    OUTPUT:
    X, Y and the columns of Y of the dataframe
    '''
    # creating engine on the database given per parameter
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    
    # reaeding data from sql table disaster_reponse
    df = pd.read_sql_table(con=engine, table_name='disaster_response')
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'])
    
    return X, Y, Y.columns


def tokenize(text):
    '''
    This function cleans the given text by tokenizing, converting it to lower case and stripping (removing leading and trailing whitespaces) 
    In addition, only those words who are not in the stopwords-list, will be added to the clean_tokens-list
    
    INPUT:
    :param text: the text to be tokenized
    
    OUTPUT:
    the cleaned tokens
    '''
    
    # tokenize the text into words
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer() # creating a lemmatizer object
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() # lemmatize, lower and strip the word
        if tok not in stopWords:
            clean_tokens.append(clean_tok) # append only to the clean_tokens-list if not a stopword

    return clean_tokens


def build_model():
    '''
    This function builds a pipline out of a featured union of a text pipeline, which contains a CountVectorizer and a Tf-IDF-Transformer
    and a text-length function which extracts the length of the given text
    As classification I decided to add the RandomForestClassifier inside the MultiOutputClassifier (which was suggested in the course)
    
    OUTPUT:
    GridSearchCV
    '''
    
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('txt_length', TextLengthExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
    ])

    parameters = {
        'clf__estimator__oob_score': (True, False),
        'clf__estimator__n_estimators': [50, 100],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This method creates a prediction and converts it into a dataframe
    Afterwards a classification report is made which compares the Y_test and Y_pred variables
    
    INPUT:
    :param model: the model to be predicted on
    :param X_test: the test variable for X
    :param Y_test: the test variable for Y
    :param category_names: the names for the columns
    '''
    
    # predicting model and converting it into a dataframe
    Y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)
    
    # creating a classification report for Y_test and Y_pred
    for col in category_names:
        print(f'Column Name:{col}\n')
        print(classification_report(Y_test[col],Y_pred[col]))


def save_model(model, model_filepath):
    '''
    This function creates a pickle file for the given model and filepath
    
    INPUT:
    :param model: the model to be written into the file
    :param model_filepath: the filepath for the pickle-file
    
    '''
    
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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