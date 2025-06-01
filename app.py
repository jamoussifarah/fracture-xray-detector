import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from fracture_detector_model import get_fracture_model

# Configuration de la page
st.set_page_config(page_title="Détecteur de Fracture", layout="centered")

# Chargement du modèle
@st.cache_resource
def load_model():
    model = get_fracture_model()
    model.load_state_dict(torch.load("saved_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Prétraitement des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Titre
st.title("🔍 Détection de fracture à partir d'une radiographie")
st.write("Charge une image de radiographie pour prédire s'il y a une fracture.")

# Chargement de l'image
uploaded_file = st.file_uploader("Glissez-déposez une image ici", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image chargée", use_container_width=True)

    # Bouton de prédiction
    if st.button("Analyser l'image"):
        # Prétraitement
        input_tensor = transform(image).unsqueeze(0)  # Shape : [1, 3, 224, 224]

        # Prédiction
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, 1)

        classes = ["Fracturé","Non fracturé"]
        result = classes[predicted_class.item()]
        score = confidence.item() * 100

        # Affichage du résultat
        if result == "Fracturé":
            st.error(f"❗ Fracture détectée avec une confiance de {score:.2f}%")
        else:
            st.success(f"✅ Aucune fracture détectée. Confiance : {score:.2f}%")
