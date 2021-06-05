# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Overview](#overview)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)


## Installation  <a name="installation"></a>

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.
There is a `requirements.txt` file which contains all required libraries.
You can install all of them by simply calling it as following:

`pip install -r requirements.txt`


## Project Overview <a name="overview"></a>

In this project, I analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages.

The csv data that is included here contains real messages that were transmitted in disaster situations. My task was to develop a machine learning algorithm that evaluates these messages and assigns them to certain categories.

This project could help several people and organizations, since they would be able to automatically categorize the messages and create new analyzes for themselves.
The code could be executed even further by connecting an API behind which the messages update automatically.

To see the classification of the messages, I included a web app execution in the code so that anyone can launch this web app locally. By entering one of the messages that are in the CSV or database, you get the corresponding categories. For example, the query would look like this:

![Overview](https://github.com/u-sahin/disaster_response_pipelines/blob/main/images/Overview.png)
![Categories](https://github.com/u-sahin/disaster_response_pipelines/blob/main/images/Categories.png)


## File Descriptions <a name="files"></a>

#### Python files
* [run.py](https://github.com/u-sahin/disaster_response_pipelines/blob/main/app/run.py)  
Running this file will open the web app on local port 3001
* [process_data.py](https://github.com/u-sahin/disaster_response_pipelines/blob/main/data/process_data.py)  
The ETL (extract, transform, load) pipeline is performed and created in this file.
* [train_classifier.py](https://github.com/u-sahin/disaster_response_pipelines/blob/main/models/train_classifier.py)  
The machine learning pipeline is set up in this file.

#### Data files
* [disaster_messages.csv](https://github.com/u-sahin/disaster_response_pipelines/blob/main/data/disaster_messages.csv)  
Contains the messages in the format `id,message,original,genre`
  * *id*: this is the ID which is later used to map if with the categories
  * *message*: the message translated into English
  * *original*: the message in its original language
  * *genre*: the genre of the message
* [disaster_categories.csv](https://github.com/u-sahin/disaster_response_pipelines/blob/main/data/disaster_categories.csv)  
Contains the categories in the format `id,categories`
  * *id*: this ID is used th map the category with the messages
  * *categories*: contains several categories seperated by `;`

#### Template files
* [master.html](https://github.com/u-sahin/disaster_response_pipelines/blob/main/app/templates/master.html)  
This files is the index-page of the web app.
By calling http://localhost:3001/, you will see the contents shown in this file.
* [go.html](https://github.com/u-sahin/disaster_response_pipelines/blob/main/app/templates/go.html)  
This files takes the master.html as a template and shows the categories and the assignment to the message given.
The query for the page looks as following: http://localhost:3001/go?query=Put+your+query+in+here

#### Database files
* [DisasterRepsonse.db](https://github.com/u-sahin/disaster_response_pipelines/blob/main/data/DisasterResponse.db)  
This database is a sample database and isn't required since you will create one yourself if you run the commands shown on the bottom.


To be able to run the web app you need to run the following commands:
* Creating the DisasterResponse.db-file:  
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
* Creating the classifier.pkl-file:  
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
* Navigate to the app folder:  
`cd app` 
* Run the web-app:  
`python run.py`

The web app will be available at http://localhost:3001/

Please be aware that the second command probably will take a few minutes, depending on the CPU power of your device.

## Results <a name="results"></a>
The main findings of the code can be found at the post available [here](https://uemitsahin.medium.com/analyzing-disaster-response-piplines-with-etl-and-ml-pipelines-a6601bab2f4d).


## Licensing, Authors, Acknowledgements, etc. <a name="licensing"></a>
Must give credit to Figure Eight for the data.
Otherwise, feel free to use the code here as you would like! 
