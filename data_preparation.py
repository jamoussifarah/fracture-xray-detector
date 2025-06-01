# data_preparation.py

import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# Chemins
CSV_PATH = "FracAtlas/dataset.csv"
IMG_DIR = "FracAtlas/images"
OUTPUT_DIR = "processed_dataset"
TRAIN_RATIO = 0.8

# Créer les dossiers nécessaires
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def prepare_dataset():
    df = pd.read_csv(CSV_PATH)

    # Filtrage pour ne garder que les images existantes
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(IMG_DIR, 'fractured', x)
        if df.loc[df['image_id'] == x, 'fractured'].values[0] == 1
        else os.path.join(IMG_DIR, 'non_fractured', x))
    df = df[df['image_path'].apply(os.path.exists)]

    # Ajouter la colonne 'label'
    df['label'] = df['fractured'].apply(lambda x: 'fractured' if x == 1 else 'non_fractured')

    # Split train/val
    train_df, val_df = train_test_split(df, test_size=1 - TRAIN_RATIO, stratify=df['label'], random_state=42)

    for subset, data in [('train', train_df), ('val', val_df)]:
        for label in ['fractured', 'non_fractured']:
            create_dir(os.path.join(OUTPUT_DIR, subset, label))

        for _, row in data.iterrows():
            src = row['image_path']
            dst = os.path.join(OUTPUT_DIR, subset, row['label'], os.path.basename(src))
            shutil.copyfile(src, dst)

    print("✅ Dataset préparé avec succès.")



if __name__ == "__main__":
    prepare_dataset(
    )
