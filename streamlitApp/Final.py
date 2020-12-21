#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import json
import re


# In[4]:


# load the sample train data
df_train_sample = pd.read_csv("./train_sample.csv")

# load column object
with open("column_list.pkl", 'rb') as output:
    column_list = pickle.load(output)

# load the texts from the json file
with open("data.json", 'r', encoding="utf8") as fp:
    text_dict = json.load(fp)

# load unique values of "primary_use" column
with open("building_primary_use_unique.pkl", 'rb') as output:
        building_primary_use_unique = pickle.load(output)
    
# load meter type dictionary object for each building
with open("building_meter_type_dict.pkl", 'rb') as output:
        building_meter_type_dict = pickle.load(output)

def final(X, labels=False):
    """
    A single function to make prediction or to compute rmsle error
    """
    # if X is a single observation, add one more dimension to match input shape
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis= 0)
    
    # convert to dataframe
    df = pd.DataFrame(X, columns= column_list)
    
    # convert columns to respective object types
    dtype_dict = {'building_id': np.int16, 
                'meter': np.int8,
                'site_id': np.int8,
                'primary_use': object,
                'square_feet': np.int32,
                'air_temperature': np.float16,
                'cloud_coverage': np.float16,
                'dew_temperature': np.float16,
                'precip_depth_1_hr': np.float16,
                'sea_level_pressure': np.float16,
                'wind_direction': np.float16,
                'wind_speed': np.float16
               }
    df = df.astype(dtype_dict)
                    
    # check if input features has right range of values
    
    # if not, return False
    if not check_input_range(df):
        return False
    
    # check if provided building has valid meter type
    if not check_buidling_meter(df):
        return False
    
    # number of provided datapoints
    n = df.shape[0]
    
    # label encoder non-numeric variable
    with open("building_primary_use_unique.pkl", 'rb') as output:
        building_primary_use_unique = pickle.load(output)
        le = LabelEncoder()
        le.fit(building_primary_use_unique)
        df.primary_use = le.transform(df.primary_use)
        
    
    # apply log1p to the area (making it more normal and dealing with extreme values)
    df.square_feet = np.log1p(df.square_feet)
    
    # change the dataformat to ease the operations
    df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d %H:%M:%S")
    
    # Add date time features by parsing timestamp
    # add dayofyear column
    df["dayofyear"] = df.timestamp.dt.dayofyear
    # add day column
    df["day"] = df.timestamp.dt.day
    # add week column
    df["weekday"] = df.timestamp.dt.weekday
    # add hour column
    df["hour"] = df.timestamp.dt.hour
    # add month column
    df["month"] = df.timestamp.dt.month
    # add weekend column
    df["weekend"] = df.timestamp.dt.weekday.apply(lambda x: 0 if x <5 else 1)
    
    # ***************************************************************************/
    # \*  Title: [3rd Place] Solution
    # \*  Author: eagle4
    # \*  Date: 2011
    # \*  Code version: N/A
    # \*  Availability: https://www.kaggle.com/c/ashrae-energy-prediction/discussion/124984
    # '''
    ##############################################################
    
    # "It is supposed to calculate the solar horizontal radiation coming into the building"
    
    latitude_dict = {0 :28.5383,
                    1 :50.9097,
                    2 :33.4255,
                    3 :38.9072,
                    4 :37.8715,
                    5 :50.9097,
                    6 :40.7128,
                    7 :45.4215,
                    8 :28.5383,
                    9 :30.2672,
                    10 :40.10677,
                    11 :45.4215,
                    12 :53.3498,
                    13 :44.9375,
                    14 :38.0293,
                    15: 40.7128}

    df['latitude'] = df['site_id'].map(latitude_dict)
    df['solarHour'] = (df['hour']-12)*15 # to be removed
    df['solarDec'] = -23.45*np.cos(np.deg2rad(360*(df['day']+10)/365)) # to be removed
    df['horizsolar'] = np.cos(np.deg2rad(df['solarHour']))*np.cos(np.deg2rad(df['solarDec']))*np.cos(np.deg2rad(df['latitude'])) + np.sin(np.deg2rad(df['solarDec']))*np.sin(np.deg2rad(df['latitude']))
    df['horizsolar'] = df['horizsolar'].apply(lambda x: 0 if x <0 else x)
    
    ##############################################################
    
    # Holiday feature
    holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
                "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
                "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
                "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
                "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
                "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
                "2019-01-01"]
    df["is_holiday"] = df.timestamp.dt.date.astype("str").isin(holidays).astype(int)
    
    # Drop redundent columns
    
    # Drop the columns which contains lots of missing values and have less or no effect on predicting the target
    drop_features = ['floor_count', 'year_built']
    df.drop(drop_features, axis=1, inplace=True)
    
    timestamp_list = df.pop("timestamp")
    
    # load the models
    with open("final_model_list.pkl", 'rb') as output:
        model_1, model_2 = pickle.load(output)

    # make prediction
    prediction_list =  (np.expm1(model_1.predict(df, num_iteration=model_1.best_iteration))* df.square_feet)/2
    prediction_list +=  (np.expm1(model_2.predict(df, num_iteration=model_2.best_iteration)) * df.square_feet)/2
    

    # If no labels are provided, return the predictions
    if labels is False:
        # rename prediction column
        df_tmp = pd.DataFrame(prediction_list.values, columns= ["Meter Reading"])
        # add timestamp columns
        df_tmp["Timestamp"] = timestamp_list
        return df_tmp

    # If labels are provided, compute evaluation metric
    RMSLE = np.sqrt(1/n * np.sum(np.square(np.log1p(prediction_list) - np.log1p(np.squeeze(labels)))))

    # return rmsle
    return RMSLE


# In[ ]:


# Overview of competition
st.title("ASHRAE - Great Energy Predictor III(The Kaggle Competition)")
st.write(text_dict["ashrae_intro_text"], unsafe_allow_html= True)


# In[ ]:


# explore more
st.write("## Check The Box To Explore More")


# In[ ]:


# Roadmap checkbox
if st.checkbox('Roadmap (Know How We Are Tackling The Problem)'):
    st.markdown(text_dict["roadmap_text"])


# In[ ]:


# Overview Dataset
if st.checkbox('Dataset (About The Dataset)'):
    st.subheader('Sample Train Data (10 Rows)')
    st.write(
    # first 10 samples
    df_train_sample.head(10)
    , text_dict["dataset_text"],"\n\n**_For EDA, Look At [This](https://drive.google.com/file/d/1rXnK9fF8E0QDBodtlGdPKENXd1Z7jjJJ/view?usp=sharing) PDF File_**."
        , unsafe_allow_html= True)


# In[ ]:


# Kaggle result
if st.checkbox('Kaggle Scoreboard (How Much Score We Got On Kaggle)'):
    st.write("# Result On kaggle Leaderboard")
    st.write('''Here we get private score of 1.322 and public score of 1.137.
    <br/>
    <img src="https://i.imgur.com/ay9j5En.png" alt="kaggle leaderboard" style="zoom:50%;" />
    ''',"**_To Know More About Experimentation And Results Read [This](https://drive.google.com/file/d/1V3yC1hH5xTv9R8cggJnO9jVLXZX_Mvta/view?usp=sharing) PDF File_**.",
             unsafe_allow_html= True)


# In[ ]:


# Overview modeling
if st.checkbox('Input Format (How To Provide Input And Get Predictions)'):
    st.subheader('Sample Train Data (10 Rows)')
    st.write(
    # first 10 samples
    df_train_sample.head(10))
    
    st.write(
        text_dict["input_format_text"], 
        unsafe_allow_html= True
    )


# In[ ]:


# Prediction And Evaluation
submit_button_object = None
if st.checkbox('Prediction And Evaluation (Input Data And Get The Predictions/ Evaluation Metric Score)'):
    st.write("## Get Prediction (No Label/s Is/Are Required) Or Compute Error Metric (Require Label/s)")
    st.write("### Enter The Input Data Here:")
    st.write(f"Column Sequence: `{np.array(column_list)}`", "\n\n**\*Note**: Not all the building has all the meter type.")
    user_input = st.text_area('', '''[789, 2, 2017-1-26 01:00:00, 7, Education, 64583, 1923.0, 1.0, -10.0, nan, -13.5, nan, 1026.0, 70.0, 4.6]''')
    submit_button_object = st.button('Make Prediction/ Compute RMSLE')
    render_graph = st.checkbox('Draw Graph (works when only features are provided)')


# In[ ]:


# convert string list to numpy array object
def string_to_list(X, y= False):
    '''Convert string list to list/numpy object'''
    # process features
    features = re.findall(r"(\[[^\[].+?\])",X)
    
    for idx, row in enumerate(features):
        # strip all spaces and convert to numpy array object
        features[idx] = np.array([element.strip() for element in row.strip('][').split(',')])
        
        # if a row doesn't have 15 features, ask user to provide it
        if len(features[idx]) != 15:
            st.error(f"Row {idx + 1} contains {len(features[idx])} features whereas there should be 15 features")
            return False
    
    features = np.array(features)
    
    # process labels and validate "labels" section
    if y:
        # parse string to list object
        labels = y.strip('][').split(',')
        
        # if we fail to convert to numpy array, invalid format provided
        try:
            labels = np.array(labels)
        except:
            # tell user to provide valid input for labels
            st.error("**Invalid label format provided**:\n\n Valid format: **[label_row_1, label_row_2, label_row_3]**")
            # return False
            return False
        
        # if non valid datatype if provided, alert the user
        try:
            labels = np.float16(labels)
        except:
            # ask the user to provide valid input format
            st.error("**Empty or Invalid Label Type Is Provided**:\n\n Valid Datatype: **Float**")
            # return False
            return False
        
        # make sure feature and labels have same 1st dimension in shape
        if labels.shape[0] != features.shape[0]:
            # alert the user to provide same number of rows
            st.error("**Number of feature rows does not match the number of label rows provided.**\n\nMake Sure To Provide Label For Each Feature Row")
            # return False
            return False
        return features, labels
    
    # return only features
    return features


# In[2]:


def check_input_range(df):
    '''Check if user inputs is in right range'''
    
    # check discrete numeric type
    # ['building_id', 'meter', 'site_id', 'primary_use']
    disc_col_dict = {0: ('building_id', [1, 1449]), 1: ('meter', [0, 3]), 3: ('site_id', [0, 15])}
    
    # for each discrete column, check the input value range
    for col_idx, disc_col_tuple in disc_col_dict.items():
        col_name = disc_col_tuple[0]
        lo = disc_col_tuple[1][0]
        hi = disc_col_tuple[1][1]
        
        if not df[col_name].between(lo, hi).all():
            st.error(f"**Value Out Of Range for column {col_idx + 1} ('{col_name}')**.\n\n Valid values should be within(inclusive): **[{lo} - {hi}]**")
            return False
    
    # check value set for primary_use column
    col_idx = 4
    col_name = 'primary_use'
    if not df[col_name].isin(building_primary_use_unique).all():
            st.error(f"**Value Out Of Range for column {4} ('{col_name}')**.\n\n Valid values should be one of :**{', '.join(building_primary_use_unique)}**")
            return False
    # all the inputs are in valid range and set, process the input further
    return True


# In[ ]:


def check_data_types(X):
    '''Check if provided input has valid data type'''
    
    dtype_dict = {0: ["building_id", "Integer",np.int16], 
                1: ['meter', "Integer", np.int8],
                3: ['site_id', "Integer", np.int8],
                5: ['square_feet', "Integer", np.int32],
                7: ['air_temperature', "Float", np.float16],
                8: ['cloud_coverage', "Float", np.float16],
                9: ['dew_temperature', "Float", np.float16],
                10: ['precip_depth_1_hr', "Float", np.float16],
                11: ['sea_level_pressure', "Float", np.float16],
                12: ['wind_direction', "Float", np.float16],
                13: ['wind_speed',"Float",  np.float16]
               }
    
    # for each column, convert to repsective data type
    for col_idx, list_ in dtype_dict.items():
        column_name, valid_type, dtype = list_[0], list_[1], list_[2]
        
        # try to convert current column into it's respective data type
        try:
            X[:, col_idx] = X[:, col_idx].astype(dtype)
        
        except Exception as e:
            
            st.error(f"**Invalid Data Type Provided For Column {col_idx} ('{column_name}')**.\n\n Valid data type: **{valid_type}**")
            return False
    
    
    # check for timestamp column
    col_idx = 2
    
    column_name, valid_type, dtype = "timestamp", "YYYY-MM-DD HH:MM:SS, eg ' 2017-12-31 01:00:00' ", "Datetime Format"
    
    # try to convert datetime column into datetime data format
    try:
        pd.to_datetime(X[:, col_idx], format="%Y-%m-%d %H:%M:%S")

    except Exception as e:
        st.error(f"**Invalid Datetime Format Provided For Column {col_idx} ('{column_name}')**.\n\n Valid Datetime Format: **{valid_type}**")
        return False
    
    return X


# In[ ]:


# check if meter type is present for a provided building
def check_buidling_meter(df):
    '''Not all the buildings has all the meter types, check if building has provided meter type'''
    
    # for each building, check the meter type
    for idx, lst in enumerate(df[["building_id", "meter"]].values):
        bid, meter = lst[0], lst[1]
        if meter not in building_meter_type_dict[bid]:
            
            # show the error
            st.error(f"**No Meter Type '{meter}' For Building {bid}**. Meter Should Be One Of: **{building_meter_type_dict[bid]}**")
            return False
    return True


# In[ ]:


# handle user submission
if submit_button_object:
    
    # if both features and labels are provided
    if len(user_input.split(";")) == 2:
         
        # divied features and labels
        X, y = user_input.split(";")
        
        # check if square brackets matches for input
        if X.count("[") != X.count("]") or y.count("[") != y.count("]"):
            # alert the user
            st.error("**Invalid Format Provided.**\n\nValid Format: **[[feature_row_1, feature_row_2, ...]];[label_row_1,  label_row_2, ...]**")
        
        else:
            # remove leading and trailing spaces
            X = X.strip()
            y = y.strip()

            # if either of the string is empty, print error message
            if not X.strip() and not y.strip():
                st.error("**Empty Input Provided**:\n\nMake sure both the inputs are non empty")
            
            # if no features are provided
            elif not X.strip():
                st.error("**Empty Input Provided For Features.**")
            
            # if no labels are provided
            elif not y.strip():
                st.error("**Empty Input Provided For Lables.**")

            # process input further
            else:
                # convert list string to numpy array
                return_value = string_to_list(X, y)

                # error has been encountered, do nothing
                if isinstance(return_value, bool):
                    # do nothing
                    pass

                # process input further
                else:
                    # Parse the input
                    X, y = return_value
                    
                    # if X is a single observation, add one more dimension to match input shape
                    if len(X.shape) == 1:
                        X = np.expand_dims(X, axis= 0)
                    
                    ##### check if input features has right datatype
                    dtype_checked = check_data_types(X)
                    
                    # if not, alert the user
                    if isinstance(dtype_checked, bool):
                        pass
                    # else process the input further
                    else:
                    
                        # compute the metric
                        RMSLE = final(X, y)

                        if isinstance(RMSLE, bool):
                            # do nothing
                            pass

                        # process the input further
                        else:
                            # update the user
                            st.write(f"# RMSLE: {RMSLE:.3f}")

    # if only features are provided
    elif len(user_input.split(";")) == 1:
        
        X = user_input
        
        # check if square brackets matches for input
        if X.count("[") != X.count("]"):
            # alert the user
            st.error("**Invalid format provided.**\n\nValid format: **[[feature_row_1, feature_row_2]];[label_row_1,  label_row_2]**")
        
        # remove leading and trailing spaces
        X = X.strip()

        # If input is empty
        if not X:
            st.error("**Empty Input Provided**:\n\nMake sure input is non empty")
        
        # process input further
        else:
            
            # convert list string to numpy array
            return_value =  string_to_list(X)

            # error has been encountered, do nothing
            if isinstance(return_value, bool):
                pass
            
            # proceed input further
            else:
                X = return_value
                
                # if X is a single observation, add one more dimension to match input shape
                if len(X.shape) == 1:
                    X = np.expand_dims(X, axis= 0)
                
                ##### check if input features has right datatype
                dtype_checked = check_data_types(X)
                    
                # if not, alert the user
                if isinstance(dtype_checked, bool):
                    pass
                
                # else process the input further
                else:
                    # else process the data further
                    predictions = final(X)

                    # error has been encountered, do nothing
                    if isinstance(predictions, bool):
                        # do nothing
                        pass
                    # process the input further
                    else:

                        st.write("# Prediction/s:")
                        st.write(predictions)

                        # if graph checkbox is checked, draw the graph
                        if render_graph:
                            # if X is a single observation, add one more dimension to the shape to match input shape
                            if len(X.shape) == 1:
                                X = np.expand_dims(X, axis= 0)

                            # render graph for each building id
                            for idx, data_list in enumerate(X[:,0:2]):

                                bid, meter = data_list

                                # load the data for building id, meter pair
                                with open(f"./building_meter_reading/meter_reading_bid_{bid}_meter_{meter}.pkl", 'rb') as output:
                                    df_meter_reading = pickle.load(output)


                                # plot all the target points of current buidling id and meter
                                fig = px.line(df_meter_reading, x="timestamp", y="meter_reading", title=f'Meter reading of building id "{bid}" and meter "{meter}" over time')

                                # st.write(pd.DataFrame(predictions.iloc[idx].values.reshape(1, 2), columns= ["Meter Reading", "Timestamp"]))
                                tmp = pd.DataFrame(predictions.iloc[idx].values.reshape(1, 2), columns= ["Meter Reading", "Timestamp"])
                                tmp["text"] = "prediction"
                                data = px.scatter(tmp,
                                                     x="Timestamp",
                                                     y="Meter Reading",
                                                 color= "Timestamp",
                                                 hover_name= "text")

                                # current prediction at given timestamp
                                fig.add_trace(data.data[0])

                                fig.update_layout(
                                autosize=False,
                                width=1000,
                                height=500,)

                                st.plotly_chart(fig)

    else:
        # invalid data format
        st.error("**Invalid Input Format Provided**.\n\nProvided format should be as below: eg **`[[789, 2, 2017-1-26 01:00:00, 7, Education, 64583, 1923.0, 1.0, -10.0, nan, -13.5, nan, 1026.0, 70.0, 4.6]];[600]`**")


# In[ ]:


# ending
st.write("<hr style='display: block;height: 1px;border: 0px ;border-top: 1px solid #ccc;margin: 1em 0;padding: 0;'>", unsafe_allow_html= True)
st.write("Have any questions or concerns? Feel free to email me at _akashkewar@gmail.com_")

