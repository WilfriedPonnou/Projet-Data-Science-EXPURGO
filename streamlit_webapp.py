import streamlit as st
import numpy as np
import pandas as pd
import folium
import cv2
import matplotlib.pyplot as plt
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
from PIL import Image
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
from datetime import datetime
from streamlit_echarts import st_echarts
from collections import Counter

st.set_option('deprecation.showfileUploaderEncoding', False)

st.set_page_config(page_title="Expurgo", page_icon="https://www.camping-croisee-chemins.fr/wp-content/uploads/2021/02/Recyclage.png")

file = './map_data.csv'
#file = r'C:\Users\Antoine\Documents\EFREI\mastercamp\projet\code38\data\map_data.csv'

locator = Nominatim(user_agent="myGeocoder")

@st.cache(suppress_st_warning=True)
def detect_objects(our_image,score_threshold,nms_threshold):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    col1, col2 = st.beta_columns(2)

    col1.subheader("Image Chargée")
    st.text("")
    plt.figure(figsize = (15,15))
    plt.imshow(our_image)
    col1.pyplot(use_column_width=True)

    # YOLO ALGORITHM
    net = cv2.dnn.readNet("custom-yolov4-detector_best.weights", "custom-yolov4-detector.cfg")

    classes = []
    with open("_darknet.labels", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0,255,size=(len(classes), 3))


    # LOAD THE IMAGE
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    height,width,channels = img.shape


    # DETECTING OBJECTS (CONVERTING INTO BLOB)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop = False)   #(image, scalefactor, size, mean(mean subtraction from each layer), swapRB(Blue to red), crop)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes =[]

    # SHOWING INFORMATION CONTAINED IN 'outs' VARIABLE ON THE SCREEN
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > score_threshold:
                # OBJECT DETECTED
                #Get the coordinates of object: center,width,height
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)  #width is the original width of image
                h = int(detection[3] * height) #height is the original height of the image

                # RECTANGLE COORDINATES
                x = int(center_x - w /2)   #Top-Left x
                y = int(center_y - h/2)   #Top-left y

                #To organize the objects in array so that we can extract them later
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)



    indexes = cv2.dnn.NMSBoxes(boxes, confidences,score_threshold,nms_threshold)
    #print(indexes)

    font = cv2.FONT_HERSHEY_SIMPLEX
    items = []
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            #To get the name of object
            label = classes[class_ids[i]]
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,7)
            items.append(label)


    st.text("")
    col2.subheader("Image avec déchets détectés")
    st.text("")
    plt.figure(figsize = (15,15))
    plt.imshow(img)
    col2.pyplot(use_column_width=True)

    if len(indexes)>1:
        st.success("Le détecteur a trouvé {} déchets - {}".format(len(indexes),[item for item in set(items)]))
    else:
        st.success("Le détecteur a trouvé {} déchet - {}".format(len(indexes),[item for item in set(items)]))
    return items

@st.cache()
def get_address(lat,lon):
    coord=str(lat)+","+str(lon)
    location = locator.reverse(coord)
    return pd.DataFrame(location.raw["address"], index=[0])

st.sidebar.subheader("Bienvenue")
a = st.sidebar.radio('Navigation:',["photo","carte","tableau de bord"])

if a == "photo":

    st.title("Détection de déchets")
    st.write("La détection de déchets a été possible grâce au dataset de TACO, au modèle Yolov4, qui conjugués, nous ont permis d'entrainer (grâce à Google Colab) notre propre modèle de détection de déchets! Ce détecteur est donc spécialisé dans la détection de déchets de tous types! Essayez par vous même 😃")
    score_threshold = st.sidebar.slider("Taux de confiance", 0.00,1.00,0.05,0.01)
    nms_threshold = st.sidebar.slider("Superposition des détectiosn(NMS)", 0.00, 1.00, 0.4, 0.01)
    uploaded_file = st.file_uploader("Uploader une image", type=['jpg','png','jpeg'])
    if uploaded_file is not None:
        our_image = Image.open(uploaded_file)
        labels=detect_objects(our_image,score_threshold,nms_threshold)

        radio_pred = [] #liste pour stocker les prédictions
        for x in labels:
            radio_pred.append(x)
        radio_pred.append("autre")
        #l'uitlisateur choisit les bonnes détection :
        pred = st.multiselect('Selectionnez les bonnes détections:', radio_pred, default=radio_pred[:-1])

        #si les prédictions sont inexactes :
        if "autre" in pred:
            with open("_darknet.labels", "r") as f:
                classes = [line.strip() for line in f.readlines()]
            true_classe = st.selectbox('Choisissez la bonne catégorie',classes[:-1])
            pred.append(true_classe.capitalize())
            pred.remove('autre')
            final_pred = pred
        else:
            final_pred = pred

        if final_pred:
            #afficher les catégories séléctionné :
            display = "     "
            for x in final_pred:
                display += x +", "
            "Dans cette image se trouve : ", display[:-2]

            #bouton valider pour récuperer la position de l'utilisateur :
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
                lat = result['GET_LOCATION']['lat']
                lon = result['GET_LOCATION']['lon']
                address = get_address(lat,lon)
                date = datetime.now().isoformat(timespec='seconds')
                final_pred = Counter(final_pred)

                #création d'un dataframe pour chaque prédiction :
                for key,value in final_pred.items():
                    label = key
                    num = value
                    new_data = pd.DataFrame({
                        'category' : label,
                        'lat' : [result['GET_LOCATION']['lat']],
                        'lon' : [result['GET_LOCATION']['lon']],
                        'date' : date,
                        'number' : num
                    })
                    new_data = pd.concat([new_data,address],axis=1)
                    map_data = pd.read_csv(file, index_col=0)
                    map_data = map_data.append(new_data, ignore_index=True)
                    map_data.to_csv(file)

                #afficher la position de l'utilisateur :
                m = folium.Map(location=[lat, lon], zoom_start=16)
                tooltip = "Votre position"
                folium.Marker(
                    [result['GET_LOCATION']['lat'], result['GET_LOCATION']['lon']], popup=final_pred, tooltip=tooltip
                ).add_to(m)
                folium_static(m)
                st.subheader("merci, le déchet a été ajouté à la carte")


if a == "carte":
    st.title("cartographie des déchets ")

    m = folium.Map(location=[46.232192999999995,2.209666999999996], zoom_start=6)
    map_data = pd.read_csv(file, delimiter=",")
    marker_cluster = MarkerCluster().add_to(m)
    #ajout des déchets sur la carte :
    for row in map_data.iterrows():
        folium.Marker(
            location=[row[1][2],row[1][3]], #position, latitude/longitude
            popup=str(row[1][1]),           #afficher la classe du déchet si on clique dessus
            icon=folium.Icon(color="red", icon="trash",prefix="fa"),
        ).add_to(marker_cluster)
    folium_static(m)


if a == "tableau de bord":
    st.title("tableau de bord")

    data = pd.read_csv(file, index_col=0)
    data = data.set_index('date')
    data.index = pd.to_datetime(data.index)

    st.sidebar.subheader('Paramètres tableau de bord:')

    #choisir la ville :
    city_list = ['Toutes les villes']+data.groupby("municipality").agg('sum').index.tolist()
    selected_city = st.sidebar.selectbox('Choisissez une ville :', city_list)
    if selected_city != 'Toutes les villes':
        data2 = data[data['municipality']==selected_city]
        data_per_classes = data2.groupby("category").agg('sum')
    else:
        data_per_classes = data.groupby("category").agg('sum')

    #choisir le type de déchet :
    waste_list = ['Tous les déchets']+data.groupby("category").agg('sum').index.tolist()
    selected_waste = st.sidebar.selectbox('Choisissez un déchet :', waste_list)
    if selected_waste != 'Tous les déchets':
        data3 = data[data['category']==selected_waste]
        data_per_city = data3.groupby("municipality").agg('sum')
        data_per_month = data3.groupby(by=[data3.index.month]).agg('sum')
    else:
        data_per_city = data.groupby("municipality").agg('sum')
        data_per_month = data.groupby(by=[data.index.month]).agg('sum')

    #affichage des graphes :
    col1, col2 = st.beta_columns(2)
    with col1:
        xAxis = data_per_city.index.tolist()
        yAxis = data_per_city["number"].tolist()
        xAxis = [x for y, x in sorted(zip(yAxis, xAxis),reverse=True)]
        yAxis.sort(reverse=True)

        st.subheader("répartition des déchets par ville :")
        #graphique en bars pour représenter le nombre de déchets par ville
        option1 = {
            "title": {
                "left": 'center',
                "text": selected_waste,
            },
            "tooltip": {"trigger": "item"},
            "dataZoom": [
                    {
                        "type": 'slider',
                        "start": 0,
                        "end": 10
                    }
                ],
            "xAxis": {
                "type": "category",
                "data": xAxis
            },
            "yAxis": {"type": "value"},
            "series": [{"data": yAxis, "type": "bar"}],
        }
        st_echarts(options=option1)

    with col2:
        st.subheader("classification des déchets par type :")

        d = []
        for i in range(len(data_per_classes['number'])):
             d.append(dict(value=int(data_per_classes['number'].iloc[i]), name=data_per_classes.index[i]))

        # graphique en donut pour représenter les différents type de déchets
        option2 = {
            "title": {
                "left": 'center',
                "text": selected_city,
            },
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
        st_echarts(options=option2)

    with col1:
        #graphique pour représenter le nombre de déchet par mois au cours de l'année
        n = len(data_per_month['number'].tolist())
        month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month = month[:n]
        st.subheader("Nombre de déchet par mois :")
        option3 = {
            "title": {
                "left": 'center',
                "text": selected_waste,
            },
            "tooltip": {"trigger": "item"},
            "xAxis": {
                "type": 'category',
                "data": month
            },
            "yAxis": {
                "type": 'value'
            },
            "series": [{
                "data": data_per_month['number'].tolist(),
                "type": 'line'
            }]
        };
        st_echarts(options=option3)

    with col2:
        st.subheader("informations :")
        debut = "Au total : "
        dechet = " déchets "
        ville = "dans toutes les villes"
        if selected_city != 'Toutes les villes':
            data = data[data['municipality']==selected_city]
            ville = "à "+selected_city
        if selected_waste != 'Tous les déchets':
            data = data[data['category']==selected_waste]
            dechet = " "+selected_waste.lower()+" "
        sum = data['number'].sum()
        st.write(debut+"**"+str(sum)+"**"+dechet+ville)

        debut = "En moyenne chaque mois : "
        dechet = " déchets "
        ville = "dans toutes les villes"
        if selected_city != 'Toutes les villes':
            data = data[data['municipality']==selected_city]
            ville = "à "+selected_city
        if selected_waste != 'Tous les déchets':
            data = data[data['category']==selected_waste]
            dechet = " "+selected_waste.lower()+" "
        data = data.groupby(by=[data.index.month]).agg('sum')
        moy = data['number'].sum()/n
        st.write(debut+"**"+str(int(moy))+"**"+dechet+ville)
        st.write("")

        st.subheader("Conseil du jour :")
        conseils = ["Choisir des lampes basse consommation","Eviter de laisser les appareils électriques en veille", "Varier la température selon les pièces","Ne pas laisser couler l'eau du robinet", "Prendre une douche plutôt qu'un bain","Imprimer en recto-verso","Choisir le covoiturage", "Se déplacer à vélo"]
        conseil = conseils[np.random.randint(len(conseils))]
        st.write(conseil)
