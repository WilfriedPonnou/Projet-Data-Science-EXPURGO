# Projet-Data-Science-EXPURGO
![Capture_d_Acran_2021-06-29_A__23](https://user-images.githubusercontent.com/69146981/124489818-b1fc8b80-ddb1-11eb-8cc6-9d2809a19940.png)
# Global description
Expurgo is a webapp wich allows to map waste thanks to trash detection model trained with taco dataset and yolov4 using google colab and roboflow for annotation.
There is an interactive map in which you can see the locations of the different trash object. Furthermore, there is a dashboard made with collected data which rises people awareness towards importance of waste collection.

# Prerequisite
python 3.8

# Installation
	git clone https://github.com/WilfriedPonnou/Projet-Data-Science-EXPURGO.git
	pip install -r requirements.txt
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12Gvkfy1AzrLOx4vR1d6wScYXJv6iBuMi' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12Gvkfy1AzrLOx4vR1d6wScYXJv6iBuMi" -O custom-yolov4-detector_best.weights && rm -rf /tmp/cookies.txt
Or download this file and add it to the repository, for the last step: https://drive.google.com/file/d/12Gvkfy1AzrLOx4vR1d6wScYXJv6iBuMi/view?usp=sharing


# Run Expurgo webapp
	streamlit run streamlit_webapp.py
Enjoy !
