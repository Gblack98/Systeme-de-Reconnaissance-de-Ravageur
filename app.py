import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tempfile
import os

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Détection de Ravageurs Agricoles",
    page_icon="🪲",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Fonction pour reconstituer le fichier
def reconstituer_fichier(parties, fichier_final):
    """
    Reconstituer le fichier final à partir des parties divisées.
    """
    if not os.path.exists(fichier_final):
        with open(fichier_final, "wb") as f:
            for partie in parties:
                with open(partie, "rb") as part_file:
                    f.write(part_file.read())
        print(f"Fichier {fichier_final} reconstitué avec succès.")
    else:
        print(f"Le fichier {fichier_final} existe déjà.")

# Charger le modèle avec mise en cache
@st.cache_resource
def load_model():
    try:
        # Liste des parties du fichier
        parties = ["modele_part_aa", "modele_part_ab", "modele_part_ac", "modele_part_ad", "modele_part_ae"]
        fichier_final = "modele_agricultural_pests (2).h5"

        # Reconstituer le fichier
        reconstituer_fichier(parties, fichier_final)

        # Charger le modèle
        model = tf.keras.models.load_model(fichier_final)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        return None

model = load_model()

# Liste des classes (à adapter selon votre modèle)
CLASSES = ['ants', 'bees', 'beetle', 'caterpillar', 'earthworms', 
           'earwig', 'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil']

# Fonction pour effectuer la prédiction
def predict_image(image):
    if image is None or image.size == 0:
        st.error("Erreur : L'image est vide ou invalide.")
        return None, None

    try:
        input_shape = model.input_shape[1:3]  # Récupérer la hauteur et largeur attendues
        img_resized = cv2.resize(image, input_shape)  # Redimensionner selon la taille attendue
        img_resized = img_resized / 255.0  # Normaliser les valeurs des pixels
        img_array = np.expand_dims(img_resized, axis=0)

        # Faire la prédiction
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        confidence = predictions[0][class_index]

        return CLASSES[class_index], confidence

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {str(e)}")
        return None, None

# Fonction pour dessiner un cadre autour du ravageur
def draw_bounding_box(image, label, confidence):
    img_with_box = image.copy()
    height, width, _ = img_with_box.shape

    # Exemple d'un cadre fictif centré
    start_point = (50, 50)
    end_point = (width - 50, height - 50)
    color = (0, 0, 255)  # Rouge (BGR)
    thickness = 5  # Épaisseur du trait augmentée

    # Dessiner le cadre
    img_with_box = cv2.rectangle(img_with_box, start_point, end_point, color, thickness)

    # Ajouter le texte
    text = f"{label} ({confidence:.2f})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5  # Taille de la police augmentée
    font_thickness = 3  # Épaisseur de la police augmentée
    text_color = (255, 255, 0)  # Jaune (BGR)
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = start_point[0] + 10
    text_y = start_point[1] - 10 if start_point[1] - 10 > text_size[1] else start_point[1] + text_size[1] + 10
    cv2.putText(img_with_box, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return img_with_box

# Interface utilisateur Streamlit
st.title("🪲 Détection de Ravageurs Agricoles")
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
    }
    .stFileUploader>div>div>button {
        background-color: #008CBA;
        color: white;
    }
    .stFileUploader>div>div>div {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.write("Téléchargez une image ou une vidéo pour identifier le type de ravageur présent.")

# Section upload d'image ou vidéo
uploaded_file = st.file_uploader("Choisissez une image ou une vidéo...", type=["jpg", "jpeg", "png", "webp", "mp4", "avi"])

if uploaded_file is not None:
    # Vérifier le type de fichier
    if uploaded_file.type.startswith('image'):
        # Traitement des images
        try:
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            # Afficher l'image avec une largeur limitée
            st.image(image, caption="Image originale", width=400)

            # Prédiction
            with st.spinner("Analyse de l'image..."):
                label, confidence = predict_image(image_np)

            if label is not None:
                # Dessiner le cadre et afficher l'image annotée
                image_with_box = draw_bounding_box(image_np, label, confidence)
                st.image(image_with_box, caption=f"Ravageur détecté : {label} ({confidence:.2f})", width=400)

                # Résultats textuels
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Ravageur détecté :** {label}")
                with col2:
                    st.info(f"**Confiance :** {confidence:.2f}")

        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image : {str(e)}")

    elif uploaded_file.type.startswith('video'):
        # Traitement des vidéos
        st.write("Analyse de la vidéo en cours...")

        # Sauvegarder la vidéo dans un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        # Lire la vidéo avec OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Erreur : Impossible d'ouvrir la vidéo. Le fichier est peut-être corrompu ou incomplet.")
        else:
            frame_placeholder = st.empty()  # Placeholder pour afficher les frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)
            frame_count = 0
            detections = {}  # Dictionnaire pour stocker les détections

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Faire une prédiction toutes les 5 frames
                if frame_count % 5 == 0:
                    label, confidence = predict_image(frame)

                    if label is not None:
                        # Dessiner le cadre sur la frame
                        frame_with_box = draw_bounding_box(frame, label, confidence)

                        # Afficher la frame annotée
                        frame_placeholder.image(frame_with_box, caption=f"Ravageur détecté : {label} ({confidence:.2f})", width=400)

                        # Enregistrer la détection
                        if label in detections:
                            detections[label] += 1
                        else:
                            detections[label] = 1

                # Mettre à jour la barre de progression
                frame_count += 1
                progress_bar.progress(frame_count / total_frames)

            cap.release()

            # Afficher un résumé des détections
            st.write("### Résumé des détections")
            for label, count in detections.items():
                st.write(f"- **{label}** : détecté {count} fois")

        os.unlink(video_path)  # Supprimer le fichier temporaire

# Bouton pour réinitialiser l'application
if st.button("Réinitialiser"):
    st.experimental_rerun()

st.write("---")
st.markdown("""
    **Application développée avec :**
    - [Streamlit](https://streamlit.io)
    """)
