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
 <p align="center">If you want to discuss more about the project, then you can reach out via <a href="mailto:idowuodesanmi@gmail.com">mail 📩</a>.</p>
<p align="center"> 
    <a href="/" target="_blank">
    <img src="https://github.com/joshasgard/Disaster-Response-App/blob/master/app/static/img/readme1.png"></img>
  </a>
  </p>


# Summary 🎯
A web app for an ML pipeline trained and deployed to help emergency responders detect public messages (tweets, FB posts, texts, etc.) asking for aid during disasters and emergencies. Victims and witnesses can also use the app to check the relevant agencies to call upon. 
<p align="center"> 
    <a href="https://github.com/joshasgard/Disaster-Response-App/blob/master/models/train_classifier.py" target="_blank">
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
| |- databoard.html # data visualizations board
| |- results.html  # classification result page of web app
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
* To run ML pipeline that trains classifier and saves. The training should take a few minutes (or hours) depending on your machine. 
```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```

##### Training Result
```
Loading data...
    DATABASE: data/DisasterResponse.db
Building model...
Training model...
...Training Time: 8112.252200126648 seconds ---
Evaluating model...
                        precision    recall  f1-score   support

               related       0.83      0.96      0.89      3992
               request       0.84      0.49      0.62       888
                 offer       0.00      0.00      0.00        25
           aid_related       0.78      0.67      0.72      2145
          medical_help       0.45      0.04      0.07       398
      medical_products       0.76      0.07      0.14       254
     search_and_rescue       0.73      0.08      0.14       139
              security       0.33      0.01      0.02       107
              military       0.38      0.02      0.04       160
           child_alone       0.00      0.00      0.00         0
                 water       0.85      0.45      0.58       331
                  food       0.84      0.61      0.71       594
               shelter       0.79      0.40      0.53       454
              clothing       1.00      0.08      0.15        97
                 money       0.60      0.03      0.05       119
        missing_people       1.00      0.01      0.03        67
              refugees       0.67      0.04      0.08       192
                 death       0.85      0.19      0.31       245
             other_aid       0.62      0.04      0.07       683
infrastructure_related       0.33      0.00      0.01       350
             transport       0.64      0.04      0.07       241
             buildings       0.75      0.16      0.26       246
           electricity       0.67      0.02      0.04       101
                 tools       0.00      0.00      0.00        30
             hospitals       0.00      0.00      0.00        51
                 shops       0.00      0.00      0.00        29
           aid_centers       0.00      0.00      0.00        72
  other_infrastructure       0.00      0.00      0.00       235
       weather_related       0.87      0.62      0.73      1459
                floods       0.90      0.32      0.48       422
                 storm       0.82      0.36      0.50       499
                  fire       0.00      0.00      0.00        57
            earthquake       0.92      0.71      0.80       514
                  cold       0.57      0.04      0.08        90
         other_weather       0.50      0.03      0.06       278
         direct_report       0.81      0.36      0.50      1009

             micro avg       0.82      0.51      0.63     16573
             macro avg       0.56      0.19      0.24     16573
          weighted avg       0.75      0.51      0.56     16573
           samples avg       0.68      0.47      0.51     16573

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
* Even with **Stratified Sampling** through Grid Search Cross Validation to maintain weights, our model still overfits (i.e. predicts 0 most of the times) for these categories. Hence, we have unreasonably high or low precisions and recalls for the classes. 

### Emphasizing Precision and/or Recall
* Precision, calculated as `TP/(TP + FP)`, is a model metric that punishes classification of messages into a category they do not belong (False Positves, returns 1 instead of 0). The higher the number of misclassifications in that category, the lower the precision value of the model for the category and vice-versa. Recall, calculated as the true positive rate `TP/(TP + FN)`, is a model performance metric that helps to measure the failure of a model to detect messages that belong to a category (False Negatives, returns 0 instead of 1). 

* We have a small positive to negative class ratio for most of the disaster category in the dataset, which supports our interest of detecting relevant messages (positive class) out of a multitude in a disaster. 

* **Hence, for each of the 36 categories, our choice of metric depends on whether it is more costly for our model to detect some wrong messages (precision) or fail to detect some messages of people that need emergency help (Recall) or both (f1 score). In this case, we can afford to detect some irrelevant messages for most categories, but we can't afford to miss messages of people that need help. **. 


# Licence 📃
* None. 

# Credits 🚀
* Data provided by <a href = https://appen.com/> FigureEight, now appen </a>
* Project reviewed by <a href = udacity.com> UDACITY </a> Data Science Team
