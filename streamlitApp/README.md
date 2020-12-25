

# Description

 Streamlit is an open source python library that is used to create data driven web app. Streamlit's beauty is in its "easy to use" library to create interactive web apps for machine learning and data science.

We have used Heroku to deploy our streamlit server. The main reason for using Heroku and not any other services is that, Heroku is really simple to use and you don't have to worry about managing infrastructure, services and so on.

To know more about deploying a python app on Heroku server have a look at [this](https://martian1231-py.medium.com/ashrae-great-energy-predictor-iii-a-machine-learning-self-case-study-5c9e9d0ea11d) post by me.

Click [here](https://ashrae-great-energy-predictor.herokuapp.com/) to see this app running on Heroku.

# Files Metadata

**Final.py**: Contains main python code as `.py` file.

**Final.ipynb**:  _jupyter_ _notebook_ version of `Final.py` file .

**data.json**: Contains large text data which is loaded onto main python file.

**column_list.pkl**: Binary file contains list of feature name in orders.

**building_primary_use_unique.pkl**: Binary file contains list of unique values which _"primary_use"_ column has.

**building_meter_type_dict.pkl**: Binary file contains dictionary object for mapping _building_id_ with _meter\_types_(eg: `{1: [0, 3], 999: [0, 1, 2, 2], ...}` which denotes "building 1 has 2 meter types, 0 and 3").

**final_model_list.pkl**: Final best trained model found after lot of experimentation and cross validation.

**Procfile**: "Procfile is a mechanism for declaring what commands are run by your application’s dynos on the Heroku platform." [Source](https://stackoverflow.com/questions/16128395/what-is-procfile-and-web-and-worker)

**requirements.txt**: Contains all the dependencies used by our python program.

**setup.sh**: Additional configuration to make streamlit work on Heroku.

**train_sample.csv**: Contains random 100 rows from train dataset.

**building_meter_reading**: Folder contains `.pkl` file for each _building_id_ and _meter_ pair. These files are used to plot the graph on streamlit app.