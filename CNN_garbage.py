import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar el uso de la GPU si está disponible
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU configurada correctamente.")
    except:
        print("No se pudo configurar la GPU.")
else:
    print("No se encontró GPU, utilizando CPU.")

# Rutas de los directorios de datos (Primero, se debe descargar el dataset desde el enlace proporcionado (https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) 
# y guardarlo en el directorio del proyecto, dentro de una carpeta llamada "Dataset". Dentro de esta carpeta, 
# deben estar organizadas las subcarpetas correspondientes, como "battery", "cardboard", etc. Luego, se ejecuta el archivo preparar_dataset.py.)
base_dir = 'garbage_dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Parámetros generales
altura_imagen, anchura_imagen = 224, 224
tamano_lote = 32
num_clases = 10  # Número de categorías en el dataset
epocas = 30

# Preprocesamiento y aumentación de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Generadores de datos
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(altura_imagen, anchura_imagen),
    batch_size=tamano_lote,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(altura_imagen, anchura_imagen),
    batch_size=tamano_lote,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(altura_imagen, anchura_imagen),
    batch_size=tamano_lote,
    class_mode='categorical',
    shuffle=False
)

# Mapear índices de clases
class_indices = train_generator.class_indices
class_names = {v: k for k, v in class_indices.items()}

# Cargar el modelo base pre-entrenado
base_model = MobileNetV2(input_shape=(altura_imagen, anchura_imagen, 3), include_top=False, weights='imagenet')

# Congelar las capas del modelo base
base_model.trainable = False

# Construir el modelo
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_clases, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Resumen del modelo
model.summary()

# Callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = callbacks.ModelCheckpoint('mejor_modelo_garbage.keras', save_best_only=True, monitor='val_loss')

# Entrenar el modelo
history = model.fit(
    train_generator,
    epochs=epocas,
    validation_data=val_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(test_generator)
print(f"Pérdida en prueba: {loss}")
print(f"Exactitud en prueba: {accuracy}")

# Guardar el modelo final
model.save('modelo_garbage_final.keras')

# Generar reporte de clasificación
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

print('Reporte de Clasificación:')
print(classification_report(test_generator.classes, y_pred, target_names=list(class_names.values())))

# Matriz de Confusión
cm = confusion_matrix(test_generator.classes, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(class_names.values()),
            yticklabels=list(class_names.values()))
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()

# Graficar resultados de entrenamiento
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Exactitud de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Exactitud de validación')
plt.xlabel('Épocas')
plt.ylabel('Exactitud')
plt.legend()
plt.title('Exactitud durante el entrenamiento')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida durante el entrenamiento')
plt.show()
