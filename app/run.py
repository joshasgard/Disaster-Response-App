import json
import plotly
import pandas as pd
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

app = Flask(__name__)

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

#   load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

#   load model
model = joblib.load("../models/classifier.pkl")


#   index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    #   Figure 1: data showing top 5 message categories in the dataset
    genre_per_category = df.iloc[:,3:].groupby('genre').sum().T
    top_category = genre_per_category.sum(axis=1).sort_values(ascending=False).reset_index()
    top_category.columns = ['categories', 'true_proportion -1']
    top_category['false_proportion -0'] = df.shape[0] - top_category['true_proportion -1']
    top_category['categories'] = top_category['categories'].apply(lambda x: str(x).replace('_', ' '))
    top_classes = top_category.head(5)

    #   Figure 2: data visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    #   Figure 3: 

    #   create visuals
    graphs = [
        {
            'data': [
                Bar(
                    name = 'Classified in class',
                    y=top_classes['categories'],
                    x=top_classes['true_proportion -1'],
                    orientation = 'h',
                    marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3))
                    
                ),
                Bar(
                    name = 'Not classified in class',
                    y=top_classes['categories'],
                    x=top_classes['false_proportion -0'],
                    orientation = 'h',
                    marker=dict(
                            color='rgba(58, 71, 80, 0.6)',
                            line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
    )
                )
            ],
            'layout':{
                'barmode' : 'stack',
                'title': 'Top 5 message categories',
                "xaxis": {
                    'title': 'Number of messages'
                },
                "yaxis": {
                    'title': 'Categories',
                    'title_standoff' : 40, 
                    'tickangle' : 45
                },
               # 'template': "seaborn"
            }

        },
       {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'template': "seaborn"
            }
        },
    ]
    
    #   encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    #   render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


#   web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

#   web page displays training data visualizations in greater detail
@app.route('/dataviz')
def dataviz():

    return render_template(
        'dataviz.html'
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)
 

if __name__ == '__main__':
    main()