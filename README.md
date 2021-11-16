<p align="center"> 
    <img src="app/static/img/Capture.PNG" align="center" height="150"></img>
</p>

<h1 align="center"> Disaster Response Relief Pipeline 🔥</h1>
<h3 align="center"> A message classifier for detecting victims of disasters or in emergency situations <br />  built with Supervised Learning and Natural Language Processing! </h3>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3810/"><img alt="Python" src="https://img.shields.io/badge/python-3.8.10-yellowgreen" /></a>
   <a href="https://pypi.org/project/Flask/2.0.1/"><img alt="Flask" src="https://img.shields.io/badge/flask-2.0.1-blue" /></a>
   <a href="https://code.jquery.com/jquery-3.3.1.min.js"><img alt="jQuery" src="https://img.shields.io/badge/jQuery-3.3.1-lightgrey" /></a>
   <a href="https://getbootstrap.com/docs/4.0/getting-started/introduction/"><img alt="BootStrap 4.0" src="https://img.shields.io/badge/BootStrap-4.0-blue" /></a>
   <a href="https://plotly.com/"><img alt="Plotly" src="https://img.shields.io/badge/plotly-get-orange" /></a> </br>
   <a href="https://scikit-learn.org/stable/install.html"><img alt="Scikit-Learn" src="https://img.shields.io/badge/sklearn-1.0.1-green" /></a>
   <a href="https://www.sqlite.org/index.html"><img alt="SQLite" src="https://img.shields.io/badge/SQLite-DB-lightgrey" /></a>
</p>
 <p align="center">If you want to discuss more about the project, then you can reach out via <a href="idowuodesanmi@gmail.com">mail 📩</a>.</p>
<p align="center"> 
    <a href="/" target="_blank">
    <img src="https://github.com/joshasgard/Disaster-Response-App/blob/master/app/static/img/readme1.png"></img>
  </a>
  </p>


# Summary 🎯
A web app for an ML pipeline trained and deployed to help emergency responders detect public messages (tweets, FB posts, texts, etc.) asking for aid during disasters and emergencies. Victims and witnesses can also use the app to check the relevant agencies to call upon. 
<p align="center"> 
    <a href="/" target="_blank">
    <img src="https://github.com/joshasgard/Disaster-Response-App/blob/master/app/static/img/MLOps_pipeline_scaling3.png"></img>
  </a>
  </p>

### Data
* The disaster response data contains tweets and text messages gathered by FigureEight Inc. after major disasters and labelled into 36 different categories of victim needs. 
* The raw data files are two `.csv` files containing the messages and categories respectively. About 26400 of the messages were given here. 

### ETL Pipeline
* The data cleaning process was done in an ETL pipeline. It involves data extraction from `.csv`, merging, and data transformation by string splitting, type casting, duplicate removal and filtering. The clean data is then loaded into an SQlite DB.

### ML Pipeline
* The ML pipeline, with a **Random Forest Classifier (RF)** as estimator, is fitted to the dataset here. 
* Our clean messages serve as the predictor variable while the 36 categories are the MultiOutput target from the pipeline. Hyperparameter Tuning using **Grid Search (with Cross Validation)** was carried out to find the hyperparameters giving the optimum model performance. 
* During the modelling process, a **Naive Bayes Classifier and an AdaBoost Classifier** were also fitted to the dataset. But there is no significant difference in their performance compared to the RF classifier. Hence, we stuck with the latter. 
* The pipeline dumps the trained model to a pickle file and returns model **Precision, Recall and Accuracy** as performance metrics. 

### Web App 
* The web app allows a user to input a single message as a sentence and the trained model returns the disaster categories the message falls with a few relevant agencies to call upon.

### Deployment
* To Heroku. *Live link will be provided later*

# File Description 📂

### Structure
```
- app
| - static
| | - img
| | | - - # all images files (.jpg, .png, and .svg)
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py  # ETL pipeline script
|- DisasterResponse.db  # database to save clean data to
|- model_metrics.csv  # trained model metrics saved for visualization

- models
|- train_classifier.py  # ML pipeline script
|- classifier.pkl  # saved model. Not available in repo

- README.md

- .gitignore
```


# Use Repo 📢✏️
Run the following commands in the project's root directory in your command line to set up your database and model.

### ETL data to SQLite DB
* To run ETL pipeline that cleans data and stores in database.
```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```
    
### Train Model in ML Pipeline
* To run ML pipeline that trains classifier and saves. The training should take a few minutes (about 5 minutes) depending on your machine. 
```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```

##### Training Result
```
Loading data...
    DATABASE: data/DisasterResponse.db
Building model...
Training model...
Evaluating model...
1 - RELATED
Precision: 0.788, Recall 0.804, Accuracy: 0.804
2 - REQUEST
Precision: 0.885, Recall 0.891, Accuracy: 0.891
3 - OFFER
Precision: 0.995, Recall 1.0, Accuracy: 0.995
4 - AID_RELATED
Precision: 0.77, Recall 0.77, Accuracy: 0.77
5 - MEDICAL_HELP
Precision: 0.913, Recall 0.928, Accuracy: 0.928
.............
.............
36 - DIRECT_REPORT
Precision: 0.853, Recall 0.862, Accuracy: 0.862
Saving model...
    MODEL: models/classifier.pkl
Trained model saved!
```

### Run web app
 * Run the following command in the app's directory to run your web app.
         `python run.py` or run with <a href="https://pypi.org/project/Flask/2.0.1/"><img alt="Flask" src="https://img.shields.io/badge/flask-2.0.1-blue" /></a>
```
C:\Desktop\Disaster-Response-App\app> python run.py
     * Serving Flask app 'run' (lazy loading)
     * Environment: development
     * Debug mode: on
     * Restarting with stat
     * Debugger is active!
     * Debugger PIN: 431-044-507
     * Running on all addresses.
       WARNING: This is a development server. Do not use it in a production deployment.
     * Running on http://192.168.43.52:3001/ (Press CTRL+C to quit)
     ......
 ```

# Imbalanced Data ⚖️
### Imbalanced Data
* The dataset is heavily imbalanced for most of the message categories, particularly important ones like `'water', 'fire', 'missing people', 'death', 'hospital', 'electricity', and 'transport'` have very few representation. This means the model performs 
* Even with **Stratified Sampling** through Grid Search Cross Validation to maintain weights, our model still overfits (i.e. predicts 0 most of the times) for these categories. Hence, we have unreasonably high *weighted* precisions and recalls/accuracies for the classes. 

### Emphasizing Precision and/or Recall
* Precision, calculated as `TP/(TP + FP)`, is a model metric that punishes classification of messages into a category they do not belong (False Positves, returns 1 instead of 0). The higher the number of misclassifications in that category, the lower the precision value of the model for the category and vice-versa. Recall, calculated as the true positive rate `TP/(TP + FN)`, is a model performance metric that helps to measure the failure of a model to detect messages that belong to a category (False Negatives, returns 0 instead of 1). 

* We have a small positive to negative class ratio for most of the disaster category in the dataset, which supports our interest of detecting relevant messages (positive class) out of a multitude in a disaster. 

* **Hence, for each of the 36 categories, our choice of metric will depend on whether it is more costly for our model to detect some wrong messages (precision) or fail to detect some messages of people that need emergency help (Recall) or both (f1 score). In this case, we can afford to detect some irrelevant messages for most categories, but we can't afford to miss messages of people that need help. Recall is more emphasised**. 


# Licence 📃
* None. 

# Credits 🚀
* Data provided by <a href = https://appen.com/> FigureEight, now appen </a>
* Project reviewed by <a href = udacity.com> UDACITY </a> Data Science Team
