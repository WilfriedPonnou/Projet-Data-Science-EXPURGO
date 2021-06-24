import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
from datetime import datetime

st.set_option('deprecation.showfileUploaderEncoding', False)

st.set_page_config(page_title="Expurgo", page_icon="https://image.flaticon.com/icons/png/512/2640/2640438.png")

file = './map_data.csv'

locator = Nominatim(user_agent="myGeocoder")

@st.cache()
def prediction(image, model):
    [shape] = model.get_layer(index=0).input_shape
    size=shape[1:-1]
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = tf.keras.preprocessing.image.smart_resize(input_arr, size)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    [predictions] = tf.keras.applications.imagenet_utils.decode_predictions(predictions, 3)
    return predictions

@st.cache()
def load_model():
    return ResNet50(weights='imagenet')

@st.cache()
def get_address(lat,lon):
    coord=str(lat)+","+str(lon)
    location = locator.reverse(coord)
    return pd.DataFrame(location.raw["address"], index=[0])

model = load_model()

a = st.sidebar.radio('Navigation:',["photo","carte"])

if a == "photo":
    st.title("uploader une photo ")
    uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image)
        predictions = prediction(image,model)
        radio_pred = []
        for x in predictions:
            [name,description,score] = x
            radio_pred.append(description)
        radio_pred.append("autre")
        pred = st.radio('Select', radio_pred)

        if pred == "autre":
            final_pred = st.text_input('Entrez la bonne catégorie')
        else:
            final_pred = pred

        "votre choix est : ", final_pred

        loc_button = Button(label="Valider")
        loc_button.js_on_event("button_click", CustomJS(code="""
            navigator.geolocation.getCurrentPosition(
                (loc) => {
                    document.dispatchEvent(new CustomEvent("GET_LOCATION", {detail: {lat: loc.coords.latitude, lon: loc.coords.longitude}}))
                }
            )
            """))
        result = streamlit_bokeh_events(
            loc_button,
            events="GET_LOCATION",
            key="get_location",
            refresh_on_update=False,
            override_height=75,
            debounce_time=0)

        if result:
            st.dataframe(result)
            lat = result['GET_LOCATION']['lat']
            lon = result['GET_LOCATION']['lon']
            address = get_address(lat,lon)
            date = datetime.now().isoformat(timespec='seconds')
            new_data = pd.DataFrame({
                'category' : [final_pred],
                'lat' : [result['GET_LOCATION']['lat']],
                'lon' : [result['GET_LOCATION']['lon']],
                'date' : date
            })
            new_data = pd.concat([new_data,address],axis=1)
            map_data = pd.read_csv(file, index_col=0)
            map_data = map_data.append(new_data, ignore_index=True)
            st.write(map_data)
            map_data.to_csv(file)
            m = folium.Map(location=[lat, lon], zoom_start=16)

            # add marker for trash
            tooltip = "Voir les déchets"
            folium.Marker(
                [result['GET_LOCATION']['lat'], result['GET_LOCATION']['lon']], popup=str(final_pred), tooltip=tooltip
            ).add_to(m)
            folium_static(m)


if a == "carte":
    st.title("cartographie des déchets ")
    map_data = pd.read_csv(file, index_col=0)
    map_data

    m = folium.Map(location=[46.232192999999995,2.209666999999996], zoom_start=6)
    map_data = pd.read_csv(file, delimiter=",")
    marker_cluster = MarkerCluster().add_to(m)
    for row in map_data.iterrows():
        folium.Marker(
            location=[row[1][2],row[1][3]],
            popup=str(row[1][1]),
            icon=folium.Icon(color="red", icon="trash",prefix="fa"),
        ).add_to(marker_cluster)
    folium_static(m)
