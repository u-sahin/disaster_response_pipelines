# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Overview](#overview)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)


## Installation  <a name="installation"></a>

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.
There is a """requirements.txt""" file which contains all required libraries.
You can install all of them by simply calling it as following:

""" pip install -r requirements.txt """


## Project Overview <a name="overview"></a>

In this project, I analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages.

The csv data that is included here contains real messages that were transmitted in disaster situations. My task was to develop a machine learning algorithm that evaluates these messages and assigns them to certain categories.

To see the classification of the messages, I included a web app execution in the code so that anyone can launch this web app locally. By entering one of the messages that are in the CSV or database, you get the corresponding categories. For example, the query would look like this:
![Overview](https://github.com/u-sahin/disaster_response_pipelines/images/overview.png)
![Categories](https://github.com/u-sahin/disaster_response_pipelines/images/categories.png)


## File Descriptions <a name="files"></a>

* [run.py](https://github.com/u-sahin/disaster_response_pipelines/app/run.py)
Running this file will open the web app on local port 3001
* [process_data.py](https://github.com/u-sahin/disaster_response_pipelines/data/process_data.py)
The ETL (extract, transform, load) pipeline is performed and created in this file.
* [train_classifier.py](https://github.com/u-sahin/disaster_response_pipelines/models/train_classifier.py)
The machine learning pipeline is set up in this file.

To be able to run the web app you need to run the following commands:
"""python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db""" 
"""python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl"""
"""cd app"""
"""python run.py"""
The web-app will be available at http://0.0.0.0:3001/

Please be aware, that the second command probably will take a few minutes, depending on the CPU power of your device.

## Results <a name="results"></a>
The main findings of the code can be found at the post available [here](https://uemitsahin.medium.com/_).


## Licensing, Authors, Acknowledgements, etc. <a name="licensing"></a>
Must give credit to Figure Eight for the data.
Otherwise, feel free to use the code here as you would like! 
