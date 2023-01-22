import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import streamlit as st
from streamlit_folium import st_folium
import folium as fl
from folium.plugins import MousePosition
#import pydeck as pdk
#import pyperclip as clp
#import easydev
#import warnings
#import lightgbm as lgb
#from lightgbm import LGBMRegressor
#from sklearn.exceptions import ConvergenceWarning
#import sklearn.metrics
#from sklearn.metrics import mean_absolute_error
#from sklearn.model_selection import cross_validate
#from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
#from sklearn.metrics import r2_score, mean_squared_error
#from sklearn import model_selection, preprocessing, metrics
import pickle

from Helpers.model_preprocessing import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

@st.cache
# Reading the necessary dataframes
def get_data():
    df = pd.read_csv("df_final.csv")
    df_city = pd.read_csv("city_center.csv") 
    return df, df_city

def main ():
    st.set_page_config(page_title="Homiday AirBNB Price Predictor", 
    page_icon="homiday_logo.png", 
    layout="wide",
    menu_items={'About':""})

    # Header Logo
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        st.write("")
    with col2:
        st.image('homiday_logo_horizontal.png')
    with col3:
        st.write("")
    
    df, df_coord = get_data()
    #st.dataframe(df.head())
    #st.dataframe(df_coord.head())

    # Defining the parameters
    sidebar = st.sidebar
    property_type = sidebar.selectbox(label="Property Type", options=['Rental unit', 'Condo', 'Home', 'Hotel', 'Loft', 'Apartment',
     'Bed and breakfast', 'Townhouse', 'Villa', 'Vacation', 'Guest suite', 'Guesthouse', 'Casa particular', 'Tiny home', 'Boat',
     'Room', 'Houseboat', 'Camper/rv', 'Cottage', ' Entire place', 'Bungalow', 'Chalet', 'Farm', 'Cabin', 'Floor', 'Pension',
     'Earthen', 'Lodge', 'Hut', 'Tent', 'Dome', 'Barn', 'Castle', 'Minsu', 'Tipi', 'Cave', 'Tower', 'Campsite', 'Windmill',
     'Island', 'Resort','Bus', 'Treehouse', 'Yurt', 'Holiday park', 'Ryokan', 'Lighthouse', 'Religious building', 'Train',
     'Container', 'Trullo', 'In-law', 'Cycladic home', 'Kezhan'])
    city = st.selectbox(label= "City", options =['Amsterdam', 'Barcelona', 'Lyon', 'Copenhagen', 'Oslo', 'Florence', 'Porto', 'Brussels', 'Berlin', 'Paris', 'Rome', 'Munich', 'Madrid', 'Stockholm', 'Prague', 'Sevilla', 'Vienna', 'Zurich'],)
    room_type = sidebar.selectbox(label= "Room Type", options =["Entire home/apt", "Private room", "Hotel room","Shared room"])
    guests = sidebar.number_input(min_value=1, max_value=16, step=1, value=1, label="Guests")
    bedrooms = sidebar.number_input(label='Bedrooms', value=1, step=1, min_value=1, max_value=21)
    beds = sidebar.number_input(label="Beds", min_value=1, max_value=40, step=1, value=1)
    bathrooms_type = sidebar.selectbox(label= "Bathroom Type", options =["Private", "Shared"])
    bathrooms = sidebar.number_input(min_value=0., max_value=20.0, step=0.5, value=1.0, label="Bathrooms")
    amenities = sidebar.multiselect(label= "Amenities", options =["Wifi", "TV", "Kitchen", "Washer", "Free parking", "Paid parking", "Air conditioning", "Dedicated workspace", "Pool", "Hot tub", "Patio", "BBQ Grill", "Outdoor dining area", "Fire pit", "Pool table", "Indoor fireplace", "Piano", "Exercise equipment", "Lake access", "Beach access", "Ski", "Outdoor shower", "Smoke alarm", "First aid kit", "Fire extinguisher", "Lock on bedroom door", "Carbon monoxide alarm" ])
    instant_bookable = sidebar.selectbox(label= "Is the place instant bookable?", help="Should guests need an approval for the reservation?", options =["Yes", "No"])
    min_night = sidebar.number_input(label='Minimum nights for the stay', help="Maximum value is 365 nights", value=1, step=1, min_value=1, max_value=365)
    max_night = sidebar.number_input(label='Maximum nights for the stay', help="Maximum value is 1125 nights", value=1, step=1, min_value=1, max_value=1125)
    availability = sidebar.slider(label='Avalability in the following 30 days', help="How many days is the place available for booking in the next 30 days ", value=1, step=1, min_value=0, max_value=30)
    cleaning_fee = sidebar.number_input(min_value=0., max_value=20.0, step=0.5, value=1.0, label="Cleaning Fee")
    security_deposit = sidebar.number_input(min_value=0., max_value=20.0, step=0.5, value=1.0, label="Security Deposit")

    def get_coord():
        lat = df_coord[df_coord["city"] == city]["lat"][df_coord[df_coord["city"] == city].index[0]]
        lon = df_coord[df_coord["city"] == city]["lon"][df_coord[df_coord["city"] == city].index[0]]
        return lat,lon
    
    def get_pos(lat,lng):
        return lat,lng

    # Defining a map at the city center
    lat,lon = get_coord()
    m = fl.Map(location=[lat,lon], 
    min_lat=df[df["city"] == city]["latitude"].min(), 
    max_lat=df[df["city"] == city]["latitude"].max(), 
    min_lon=df[df["city"] == city]["longitude"].min(), 
    max_lon=df[df["city"] == city]["longitude"].max(), 
    max_bounds=True,
    min_zoom=10, 
    max_zoom=18, 
    zoom_start=10)

    m.add_child(fl.LatLngPopup())
    m.add_child(fl.ClickForLatLng(alert=False))
    
    map1 = st_folium(m, height=350, width=700)
    
    latitude, longitude = get_pos(map1['last_clicked']['lat'],map1['last_clicked']['lng'])
    data1 = (latitude,longitude)

    formatter = "function(num) {return L.Util.formatNum(num, 3) + ' ° ';};"

    MousePosition(
    position="topright",
    separator=" | ",
    empty_string="NaN",
    lng_first=True,
    num_digits=20,
    prefix="Coordinates:",
    lat_formatter=formatter,
    lng_formatter=formatter).add_to(m)
   
    start_cont = st.container()
    start_cont.title('🏘️ Welcome to the AirBnb Price Predictor')
    start_cont.markdown( 'Give the properties of your house, find out at what price you can advertise.')

    # Defining the price prediction model and button
    if st.button(label = 'Get Recommended Price', type='primary'):

        main_df = df.drop("price", axis=1)
        
        def instant_bookable_return():
            if instant_bookable == "Yes":
                return "t"
            else:
                return "f"
        
        data = [{'host_response_rate': main_df ['host_response_rate'].median(),
        'host_acceptance_rate': main_df ['host_acceptance_rate'].median(),
        'host_listings_count': main_df ['host_listings_count'].median(),
        'host_total_listings_count': main_df ['host_total_listings_count'].median(),
        'latitude': latitude,
        'longitude': longitude,
        'accommodates': guests,
        'bathrooms': bathrooms,
        'bathrooms_type': bathrooms_type.lower(),
        'bedrooms': bedrooms,
        'beds': beds,
        'minimum_nights_avg_ntm': min_night,
        'maximum_nights_avg_ntm': max_night,
        'availability_30': availability,
        'availability_60': availability*2,
        'availability_90': availability*3,
        'availability_365': availability*12,
        'number_of_reviews': main_df ["number_of_reviews"].median(),
        'number_of_reviews_ltm': main_df ["number_of_reviews_ltm"].median(),
        'review_scores_rating': main_df ["review_scores_rating"].mean(),
        'review_scores_accuracy': main_df ["review_scores_accuracy"].mean(),
        'review_scores_cleanliness': main_df ["review_scores_cleanliness"].mean(),
        'review_scores_checkin': main_df ["review_scores_checkin"].mean(),
        'review_scores_communication': main_df ["review_scores_communication"].mean(),
        'review_scores_location': main_df ["review_scores_location"].mean(),
        'review_scores_value': main_df ["review_scores_value"].mean(),
        'calculated_host_listings_count': main_df ["calculated_host_listings_count"].median(),
        'reviews_per_month': main_df ["reviews_per_month"].median(),
        'property_type': property_type.lower(),
        'room_type': room_type,
        'instant_bookable': instant_bookable_return(),
        'amenities': amenities,
        'city': city}]
        
        features = pd.DataFrame(data)

        final_data = model_preprocessing(features, df_coord, main_df)
        
        load_model = pickle.load(open('model.pkl','rb'))
        prediction = load_model.predict(final_data)

        st.header("Rental Features")
        col1, col2, col3 = st.columns(3)
        col1.subheader(f"City: {city}")
        col2.subheader(f"Property Type: {property_type}")
        col3.subheader(f"Room Type: {room_type}")
        col1.subheader(f"Bedrooms: {bedrooms}")
        col2.subheader(f"Bathrooms: {bathrooms} {bathrooms_type} bathroom(s)")
        col3.subheader("Amenities: "+f"{amenities}".strip("[]"))

        st.write("----------------")
        
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        col1.metric(label="Rental Price", value=(str(int(prediction))+"€/day"))
        col2.title("+")
        col3.metric(label="Cleaning Fee", value=(str(cleaning_fee)+"€"))
        col4.title("+")
        col5.metric(label="Security Deposit", value=(str(security_deposit)+"€"))
        col6.title("=")
        col7.metric(label="Total Price", value=(str(int(prediction) + cleaning_fee + security_deposit)+"€"))

        st.write("----------------")
        st.map(features[["latitude","longitude"]])
main()
