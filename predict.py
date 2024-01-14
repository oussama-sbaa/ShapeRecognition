import numpy as np
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from extra_keras_datasets import emnist
import matplotlib.pyplot as plt
import os

def label_to_char(label):
    if label < 26:
        return chr(label + 65)  # Majuscules (A-Z)
    else:
        return chr(label + 71)  # Minuscules (a-z)

# Chargement d'une image depuis EMNIST
(train_images, _), _ = emnist.load_data(type='letters')
image_index = np.random.randint(0, len(train_images))  # Choix aléatoire d'une image
image = train_images[image_index]
label = label_to_char(image_index % 52)  # Génération de l'étiquette correspondante

# Prétraitement de l'image
preprocessed_image = image.astype('float32') / 255.0
preprocessed_image = np.expand_dims(preprocessed_image, axis=-1)
preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

# Chargement du meilleur modèle
best_model = models.load_model('models/best_model.keras')

# Prédiction
prediction = best_model.predict(preprocessed_image)
predicted_label = np.argmax(prediction)
predicted_char = label_to_char(predicted_label)

# Affichage du caractère prédit
print(f"Texte extrait de l'image : {predicted_char}")

# Sauvegarde de l'image avec le texte prédit
output_folder = 'predicted_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

plt.imshow(image, cmap='gray')
plt.title(f"Prédiction : {predicted_char}")
plt.axis('off')
plt.savefig(f"{output_folder}/predicted_{predicted_char}.png")
