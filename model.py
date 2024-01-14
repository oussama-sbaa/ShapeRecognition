import numpy as np
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from extra_keras_datasets import emnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Chargement des données EMNIST
(train_images, train_labels), (test_images, test_labels) = emnist.load_data(type='letters')

# Prétraitement des données
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# Encodage One-Hot des étiquettes
train_labels = to_categorical(train_labels - 1, 52)
test_labels = to_categorical(test_labels - 1, 52)

# Augmentation des données
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)

# Modèle CNN plus complexe
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(52, activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Définition du chemin pour sauvegarder le meilleur modèle
checkpoint_path = 'models/best_model.keras'

# Entraînement du modèle
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=32),
    epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max')]
)

# Chargement et évaluation du meilleur modèle
best_model = models.load_model(checkpoint_path)
test_loss, test_accuracy = best_model.evaluate(test_images, test_labels)
print(f"Précision sur l'ensemble de test : {test_accuracy}")
print(f"Perte sur l'ensemble de test : {test_loss}")
