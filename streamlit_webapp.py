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
from streamlit_echarts import st_echarts

st.set_option('deprecation.showfileUploaderEncoding', False)

st.set_page_config(page_title="Expurgo", page_icon="https://www.camping-croisee-chemins.fr/wp-content/uploads/2021/02/Recyclage.png")

file = './data/map_data.csv'

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

st.sidebar.subheader("Bienvenue")
a = st.sidebar.radio('Navigation:',["photo","carte","tableau de bord"])

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
        num = st.number_input('Enter a number', min_value=0,value=1)

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
                'date' : date,
                'number' : num
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
            st.subheader("merci, le déchet a été ajouté à la carte")


if a == "carte":
    st.title("cartographie des déchets ")
    map_data = pd.read_csv(file, index_col=0)

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

if a == "tableau de bord":
    st.title("tableau de bord")

    data = pd.read_csv(file, index_col=0)

    st.sidebar.subheader('Paramètres tableau de bord:')

    city_list = ['all cities']+data.groupby("town").agg('sum').index.tolist()
    selected_city = st.sidebar.selectbox('Select your city :', city_list)
    if selected_city != 'all cities':
        data2 = data[data['town']==selected_city]
        data_per_classes = data2.groupby("category").agg('sum')
    else:
        data_per_classes = data.groupby("category").agg('sum')

    waste_list = ['all waste']+data.groupby("category").agg('sum').index.tolist()
    selected_waste = st.sidebar.selectbox('Select a waste :', waste_list)
    if selected_waste != 'all waste':
        data2 = data[data['category']==selected_waste]
        data_per_city = data2.groupby("town").agg('sum')
    else:
        data_per_city = data.groupby("town").agg('sum')

    col1, col2 = st.beta_columns(2)
    with col1:
        xAxis = data_per_city.index.tolist()
        yAxis = data_per_city["number"].tolist()
        xAxis = [x for y, x in sorted(zip(yAxis, xAxis),reverse=True)]
        yAxis.sort(reverse=True)

        st.subheader("répartition des déchets par ville :")
        options = {
            "dataZoom": [
                    {
                        "type": 'slider',
                        "start": 0,
                        "end": 50
                    }
                ],
            "xAxis": {
                "type": "category",
                "data": xAxis
            },
            "yAxis": {"type": "value"},
            "series": [{"data": yAxis, "type": "bar"}],
        }
        st_echarts(options=options)

    with col2:
        st.subheader("classification des déchets par type :")

        d = []
        for i in range(len(data_per_classes['number'])):
            d.append(dict(value=int(data_per_classes['number'].iloc[i]), name=data_per_classes.index[i]))

        options = {
            "tooltip": {"trigger": "item"},
            #"legend": {"left": "center"},
            "series": [
                {
                    "name": "nombre de déchet :",
                    "type": "pie",
                    "radius": ["40%", "70%"],
                    "avoidLabelOverlap": False,
                    "itemStyle": {
                        "borderRadius": 10,
                        "borderColor": "#fff",
                        "borderWidth": 2,
                    },
                    "label": {"show": False, "position": "center"},
                    "emphasis": {
                        "label": {"show": True, "fontSize": "20", "fontWeight": "bold"}
                    },
                    "labelLine": {"show": False},
                    "data": d,
                }
            ],
        }
        st_echarts(options=options)
