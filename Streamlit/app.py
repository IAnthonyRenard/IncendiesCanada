import streamlit as st 
import pandas as pd
import numpy as np
import glob
import os
import cv2
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.model_selection import train_test_split
from keras.models import load_model
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from random import randint
import seaborn as sns



st.set_option('deprecation.showPyplotGlobalUse', False)




def main():    #Fonction principale qui contient l'application
      
    st.title("Application IA pour la détection de zones à risques d'incendie au Canada")
    st.subheader("Auteur : Anthony RENARD")
    
    # Fonction d'importation des données
    #@st.cache_data(persist=True)
    def load_data():
        
        #Sous forme de np_array
        
        #Lien pour fonctionnement local
        #folder_path1 = "../input/fire_prediction_images_no_yes"
        #folder_path2 = "fire_prediction_images_no_yes2"
        
        #Lien pour Streamlit
        folder_path1 = "INPUT/fire_prediction_images_no_yes"
        folder_path2 = "Streamlit/fire_prediction_images_no_yes2"
        
        
        no_images = os.listdir(folder_path2 + '/no/')
        yes_images = os.listdir(folder_path2 + '/yes/')
        dataset=[]
        lab=[]
        
        for image_name in no_images:
            image=cv2.imread(folder_path2 + '/no/' + image_name)
            image=Image.fromarray(image,'RGB')
            image=image.resize((120,120))#240,240
            dataset.append(np.array(image))
            lab.append(0)
            
        for image_name in yes_images:
            image=cv2.imread(folder_path2 + '/yes/' + image_name)
            image=Image.fromarray(image,'RGB')
            image=image.resize((120,120))#240,240
            if image is None:
                print(image_name)
            dataset.append(np.array(image))
            lab.append(1)
        
        dataset = np.array(dataset)
        lab = np.array(lab)
        
        
        #Sous forme de DataFrame
        path = folder_path1
        path_imgs = list(glob.glob(path+'/**/*.jpg'))
        labels = list(map(lambda x:os.path.split(os.path.split(x)[0])[1], path_imgs))
        file_path = pd.Series(path_imgs, name='File_Path').astype(str)
        labels = pd.Series(labels, name='Labels')
        data = pd.concat([file_path, labels], axis=1)
        data = data.sample(frac=1).reset_index(drop=True)
        print(data.shape)
        
        return data, dataset, lab
    
    #Importation des données
    data, dataset, lab = load_data()
    #-------------------------------------------------------------------------------------------------------------------------
    
    #Affichage de la DataFrame
    if st.sidebar.checkbox("Afficher la liste des photos et leur label",False): #False = décoché par défaut
        st.subheader("Jeu de données illustrant les photos disponibles (crée à partir du dossier contenant les photos)")
        st.write(data)
    #-------------------------------------------------------------------------------------------------------------------------
    #Split des données
    
    #@st.cache_data(persist=True)
    def split(data, dataset, lab) :
            
        #Split des data sous forme de np.array
        x_train, x_test, y_train, y_test = train_test_split(dataset, lab, stratify=lab, test_size=0.2, shuffle=True, random_state=42)
        
        #Split des data sous forme de DateFrame
        train_df, test_df = train_test_split(data, stratify=data["Labels"], test_size=0.2, shuffle=True, random_state=42)
        
        return x_train, x_test, y_train, y_test, train_df, test_df
    
    
    #x_train, x_test, y_train, y_test, train_df, test_df=split(data, dataset, lab)
     #-------------------------------------------------------------------------------------------------------------------------
    
    # ViT
    
    class Patches(layers.Layer):
        def __init__(self, patch_size):
            super(Patches, self).__init__()
            self.patch_size = patch_size

        def call(self, images):
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches
    
    class PatchEncoder(tf.keras.layers.Layer):
        def __init__(self, num_patches, projection_dim):
            super(PatchEncoder, self).__init__()
            self.num_patches = num_patches
            self.projection = layers.Dense(units=projection_dim)
            self.position_embedding = layers.Embedding(
                input_dim=num_patches, output_dim=projection_dim
            )

        def call(self, patch):
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            encoded = self.projection(patch) + self.position_embedding(positions)
            return encoded
    
 
     #-------------------------------------------------------------------------------------------------------------------------    
    #Affichage d'un échantillon de photos
    st.subheader("Affichage de quelques photos du dataset")
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15),
                        subplot_kw={'xticks': [], 'yticks': []})
    for i, ax1 in enumerate(axes.flat):
        j=randint(0,len(data)-1)# Création de la variable j pour afficher des photos de manière aléatoire
        ax1.imshow(plt.imread(data.File_Path[j]))
        ax1.set_title(data.Labels[j])
        ax1.title.set_size(25)
    #plt.tight_layout()
    st.pyplot(fig)
    
    
    #Affichage de la localisation de chaque photo sur une carte
    st.subheader("Localisation des photos")
    
    #data_geo=pd.read_csv("df_geo.csv")#lien local
    data_geo=pd.read_csv("Streamlit/df_geo.csv")#lien Streamlit
    
    list_color=data_geo['Labels'].tolist()
    st.map(data_geo,latitude='latitude', longitude='longitude', color='Labels')
    
    #Affichage de la distribution des photos en 2 classes
    st.subheader("Distribution des photos (quantité)")
    fig2, ax2 = plt.subplots()
    counts = data.Labels.value_counts(normalize=False)
    cols = ['#FA3E0C' if (x =="yes") else '#17FA2B' for x in data.Labels]
    ax2 = sns.barplot(x=counts.index, y=counts, palette=cols)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=50);
    ax2.bar_label(ax2.containers[0], fmt='%.0f')
    st.pyplot(fig2)
    
    #Affichage de la distribution des photos en 2 classes %
    st.subheader("Distribution des photos (%)")
    fig3, ax3 = plt.subplots()
    counts = data.Labels.value_counts(normalize=True)
    cols = ['#FA3E0C' if (x =="yes") else '#17FA2B' for x in data.Labels]
    ax3 = sns.barplot(x=counts.index, y=counts, palette=cols)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=50);
    ax3.bar_label(ax3.containers[0], fmt='%.2f')
    st.pyplot(fig3)
    
    
    
    ######################################################################################### 
    st.subheader("Affichage de quelques transformations")
    
    def fct_transformation(dataset):
        j=randint(0,len(dataset)-1)
        plt.figure(figsize= (10,10))
        st.markdown("Voici la photo sélectionnée")
        st.image(dataset[j],width=120)
        
        from scipy.ndimage import rotate
        from scipy.ndimage import gaussian_filter
        import numpy.fft
        from scipy import ndimage
        from scipy.ndimage import shift
        
        #Rotation
        st.markdown("Rotation de la photo")
        rotate_img = rotate(dataset[j], 45)
        st.image(rotate_img)
        
        #Zoom
        st.markdown("Zoom sur la photo")
        # Création d'une image PILLOW à partir de l'image en array
        pil_image = Image.fromarray(np.uint8(dataset[j]))
        #Facteur pour zoomer
        zoom_factor = 2
        # Calcul de la nouvelle dimension
        new_width = dataset[j].shape[1] * zoom_factor
        new_height = dataset[j].shape[0] * zoom_factor
        # Resize avec Pilow pour garder le nombre de channel identique (3)
        zoomed_pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        # Convertion du zoom en array
        zoomed_image = np.array(zoomed_pil_image)
        st.image(zoomed_image)
        #Remarque : l'utilisation de scipy.ndimage.zoom ne permet pas de maintenir directement le channel RGB à 3
        
        #Filtre Gaussien
        st.markdown("Filtre Gaussien")
        gauss_img =  gaussian_filter(dataset[j],2)
        st.image(gauss_img)
        
        #Shift Fourrier
        st.markdown("Shift Fourrier")
        fig, (ax1, ax2) = plt.subplots(1, 2)
        #plt.gray()  
        input_ = numpy.fft.fft2(dataset[j])
        result = ndimage.fourier_shift(input_, shift=5)
        result = numpy.fft.ifft2(result)
        #ax1.imshow(dataset[j].astype('uint8'))
        st.image(result.real.astype('uint8'))
        
        #Shift
        st.markdown("Shift")
        shift_img =  shift(dataset[j],1)
        st.image(shift_img)


    
    
    if st.button("Lancer des transformations sur une image aléatoire"):
        fct_transformation(dataset)
        
    ######################################################################################### 


    st.subheader("Prédiction avec le modèle ViT")
    
    def chargement_model(): 
        #Chargement du modèle ViT
        
        #Lien local
        #bestModel=load_model('../modeles/fire_detection_120_120/ViT_fire_120_120_b300_e200.h5', custom_objects={'Patches': Patches,'PatchEncoder': PatchEncoder})
        #Lien Streamlit
        bestModel=load_model('modeles/fire_detection_120_120/ViT_fire_120_120_b300_e200.h5', custom_objects={'Patches': Patches,'PatchEncoder': PatchEncoder})
        
        
        #results = bestModel.evaluate(x_test, y_test)
        st.markdown("Modèle ViT chargé")
        #st.markdown('Le modèle atteint un niveau de "accuracy" de {}%!'.format(round(results[1]*100,2)))
        return bestModel

     
  

    
    def fct_prediction(x_test, y_test):
        
        st.markdown("Chargement du modèle ViT = 25s")
        st.markdown("Prédiction = 1s")
        model=chargement_model()
    
        j=randint(0,len(y_test)-1)
        img = np.expand_dims(x_test[j], axis=0)
        #j=762
        if y_test[j]==0:
            st.markdown("Il n'y a pas de risque d'incendie sur la zone suivante :")
        else :
            st.markdown("Il y a risque d'incendie sur la zone suivante :")
            
        plt.figure(figsize= (10,10))
        st.image(x_test[j],width=120)
        
        
        prediction=model.predict(img)
        pred=np.argmax(prediction)
        if pred==1:
            st.markdown("Le modèle ViT prédit un risque d'incendie dans cette zone")
        else :
            st.markdown("Le modèle ViT ne prédit pas un risque d'incendie dans cette zone")
        
        if pred==y_test[j]:
            st.markdown("Prédiction OK")
        else :
            st.markdown("Prédiction NOK")
    
    if st.button("Option 1 : lancer la prédiction sur une image aléatoire"):
        #fct_prediction(x_test, y_test)
        fct_prediction(dataset, lab)
        
    
     #########################################################################################  
    #@st.cache_data(persist=True)
    def fct_prediction2(img):
        
        st.markdown("Chargement du modèle ViT = 25s")
        st.markdown("Prédiction = 1s")
        model=chargement_model()
        prediction=model.predict(img)
        pred=np.argmax(prediction)
        if pred==1:
            st.markdown("Le modèle ViT prédit un risque d'incendie dans cette zone")
        else :
            st.markdown("Le modèle ViT ne prédit pas un risque d'incendie dans cette zone")


    st.markdown("Option 2 : lancer la prédiction sur une image choisie")       
            
    # Charger l'image depuis st.file_uploader
    uploaded_file = st.file_uploader("Télécharger une photo satellite...", type=['png', 'jpeg', 'jpg'])

    if uploaded_file is not None:
        # Ouvrir l'image avec PIL
        image = Image.open(uploaded_file)

        # Redimensionner l'image en 120x120 pixels
        image = image.resize((120, 120))
    
        # Convertir l'image en tableau NumPy
        image_array = np.array(image)
        
         # Ajouter une dimension pour obtenir un tableau de taille (None, 120, 120, 3)
        image_array = np.expand_dims(image_array, axis=0)

        # Vous pouvez maintenant travailler avec l'image sous forme de tableau NumPy
        st.image(image, caption='Image téléchargée', use_column_width=True)
    
        fct_prediction2(image_array)
        
        
if __name__=='__main__':
    main()
    