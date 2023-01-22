import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder

def model_preprocessing(dataframe, dataframe_city, dataframe_main):
    dataframe['NEW_amenities_length'] = dataframe['amenities'].apply(lambda x: len(x))

    dataframe.drop("amenities", axis=1, inplace=True)

    def grab_col_names(dataframe, cat_th=10, car_th=135):
        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]
        
        # num_cols
        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        return cat_cols, num_cols, cat_but_car
    
    df_merged = pd.merge(dataframe, dataframe_city, on='city',how="left")

    from math import radians, degrees, sin, cos, asin, acos, sqrt
    def great_circle(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        return 6371 * (acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2)))
    
    df_merged['center_distance'] = df_merged.apply(lambda row: great_circle(row['longitude'], row['latitude'], row['lon'], row['lat']),axis=1)
    dataframe['NEW_center_distance'] = df_merged['center_distance']
    bins = [-1, 1, 2, 4, 8, 15,30, 1200]
    category = ['cok yakin', 'yakin', 'biraz yakin', "orta", "biraz uzak", "uzak", "cok uzak"]
    dataframe['NEW_center_distance_cat'] = pd.cut(dataframe['NEW_center_distance'], bins, labels = category)
    dataframe['NEW_center_distance_cat'] = dataframe['NEW_center_distance_cat'].astype("O")

    dataframe['NEW_totalrooms'] = dataframe['bedrooms'] + dataframe['bathrooms']
    dataframe["NEW_review_avg"] = (dataframe["review_scores_rating"] + dataframe["review_scores_accuracy"] + dataframe["review_scores_cleanliness"] + dataframe["review_scores_checkin"] + dataframe["review_scores_communication"] + dataframe["review_scores_location"] + dataframe["review_scores_value"]) / 7

    all_data = pd.concat([dataframe, dataframe_main], axis=0)
    
    mask = all_data["property_type"].map(all_data["property_type"].value_counts()< 3957) 
    all_data["property_type"] =  all_data["property_type"].mask(mask, 'Other')
    
    mask = all_data["room_type"].map(all_data["room_type"].value_counts()) < 3000
    all_data["room_type"] =  all_data["room_type"].mask(mask, 'Other')

    cat_cols, num_cols, cat_but_car = grab_col_names(all_data,car_th=1000)
    
    binary_cols = [col for col in all_data.columns if all_data[col].dtypes == "O" and len(all_data[col].unique()) == 2]

    def label_encoder(dataframe, binary_col):
        labelencoder = LabelEncoder()
        dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
        return dataframe
    
    for col in binary_cols:
        label_encoder(all_data, col)
        
    def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
        dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
        return dataframe
    
    all_data = one_hot_encoder(all_data, cat_cols, drop_first=True)

    all_data = all_data[:1]

    return all_data
