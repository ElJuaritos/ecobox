import os
import shutil
import random

# Rutas
original_dataset_dir = 'Dataset'
base_dir = 'garbage_dataset'

# Crear carpetas principales
os.makedirs(base_dir, exist_ok=True)
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
val_dir = os.path.join(base_dir, 'val')
os.makedirs(val_dir, exist_ok=True)
test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

# Listar las clases
clases = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']

for clase in clases:
    # Crear carpetas para cada conjunto
    os.makedirs(os.path.join(train_dir, clase), exist_ok=True)
    os.makedirs(os.path.join(val_dir, clase), exist_ok=True)
    os.makedirs(os.path.join(test_dir, clase), exist_ok=True)
    
    # Obtener rutas de las imágenes
    dir_clase = os.path.join(original_dataset_dir, clase)
    imagenes = os.listdir(dir_clase)
    random.shuffle(imagenes)
    
    # Calcular índices para división
    total = len(imagenes)
    train_end = int(0.7 * total)
    val_end = int(0.85 * total)
    
    # Dividir imágenes
    train_imagenes = imagenes[:train_end]
    val_imagenes = imagenes[train_end:val_end]
    test_imagenes = imagenes[val_end:]
    
    # Copiar imágenes a las carpetas correspondientes
    for img in train_imagenes:
        src = os.path.join(dir_clase, img)
        dst = os.path.join(train_dir, clase, img)
        shutil.copyfile(src, dst)
    
    for img in val_imagenes:
        src = os.path.join(dir_clase, img)
        dst = os.path.join(val_dir, clase, img)
        shutil.copyfile(src, dst)
    
    for img in test_imagenes:
        src = os.path.join(dir_clase, img)
        dst = os.path.join(test_dir, clase, img)
        shutil.copyfile(src, dst)
