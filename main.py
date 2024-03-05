import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import shutil 

def load_images(directory, target_size=(224, 224)):
    images = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename))
        if img is not None:
            img = cv2.resize(img, target_size)
            images.append(img)
    return np.array(images)

if __name__ == "__main__":
    images_directory = "./dataset"
    images = load_images(images_directory)

    # Ustaw ResNet50 bez ostatniej warstwy
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    # Przetwórz obrazy i uzyskaj cechy
    images_resized = [cv2.resize(img, (224, 224)) for img in images]
    images_preprocessed = preprocess_input(np.array(images_resized).astype('float32'))
    features = model.predict(images_preprocessed)

    # Spłaszcz dane do jednego wymiaru
    flattened_features = features.reshape((features.shape[0], -1))

    # Ustaw k-means na spłaszczonych cechach
    kmeans = KMeans(n_clusters=8, random_state=42)
    clusters = kmeans.fit_predict(flattened_features)

    # Znajdź obrazy najbardziej oddalone od centrów klastrów (outliery)
    distances_to_centers = np.min(np.square(flattened_features[:, np.newaxis] - kmeans.cluster_centers_), axis=2)
    outlier_indices = np.argsort(distances_to_centers.sum(axis=1))[-4:]

    # Stwórz katalog dla outlayerów
    os.makedirs('./outliers', exist_ok=True)

    for idx in outlier_indices:
        cv2.imwrite(f"./outliers/{idx}_Outlier.png", images[idx])

    # Stwórz katalogi dla każdej kategorii
    for i in range(8):
        os.makedirs(f'./categories/category_{i}', exist_ok=True)

    # Przyporządkuj pozostałe obrazy do kategorii
    for idx, cluster_idx in enumerate(clusters):
        if idx not in outlier_indices:
            category_folder = os.path.join('./categories', f'category_{cluster_idx}')
            cv2.imwrite(f"{category_folder}/{idx}_Image.png", images[idx])

    # Sprawdź kategorie z jednym zdjęciem i dodaj je do outlayerów
    for i in range(8):
        category_folder = os.path.join('./categories', f'category_{i}')
        category_images = os.listdir(category_folder)

        if len(category_images) == 1:
            image_idx = int(category_images[0].split('_')[0])
            shutil.move(os.path.join(category_folder, category_images[0]), f'./outliers/{image_idx}_Outlier.png')
            os.rmdir(category_folder)
