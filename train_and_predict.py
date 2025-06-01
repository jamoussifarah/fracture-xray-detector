# train_and_predict.py

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fracture_detector_model import get_fracture_model
import os
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Paramètres
data_dir = "processed_dataset"
batch_size = 32
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vérifier si les dossiers existent
train_path = os.path.join(data_dir, 'train')
val_path = os.path.join(data_dir, 'val')

if not os.path.exists(train_path) or not os.path.exists(val_path):
    raise FileNotFoundError(f"Les dossiers 'train' ou 'val' sont introuvables dans {data_dir}. Vérifiez votre structure.")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Chargement des données
try:
    train_loader = DataLoader(
        datasets.ImageFolder(train_path, transform=transform),
        batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(
        datasets.ImageFolder(val_path, transform=transform),
        batch_size=batch_size, shuffle=False)
    
    print("✅ Chargement des données réussi.")

except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement des images : {e}")

# Vérification rapide des images
for class_folder in ['fractured', 'non_fractured']:
    folder = os.path.join(train_path, class_folder)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Le dossier {folder} est manquant.")
    for fname in os.listdir(folder)[:3]:
        try:
            img = Image.open(os.path.join(folder, fname))
            img.verify()
        except Exception as e:
            print(f"⚠️ Erreur avec l'image {fname} : {e}")

# Modèle
model = get_fracture_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Boucle d'entraînement
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Époque {epoch+1}/{num_epochs} - Batch {batch_idx} - Loss: {loss.item():.4f}")

    print(f"✅ Fin de l'époque {epoch+1} - Loss total: {total_loss:.4f}")

# Évaluation
model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f"✅ Précision sur validation : {accuracy:.2f}%")
torch.save(model.state_dict(), "saved_model.pth")
