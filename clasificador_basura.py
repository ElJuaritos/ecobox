import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import matplotlib.pyplot as plt
import cv2
import threading

class ClasificadorBasuraGUI:
    def __init__(self, master):
        self.master = master
        master.title("Clasificador de Residuos")
        master.geometry("800x600")
        master.resizable(True, True)
        master.configure(background='#f0f0f0')  # Color de fondo

        # Estilo personalizado
        self.estilo = ttk.Style()
        self.estilo.theme_use('clam')
        self.estilo.configure('TFrame', background='#f0f0f0')
        self.estilo.configure('TButton', font=('Arial', 12, 'bold'), foreground='#ffffff', background='#0078D7')
        self.estilo.map('TButton', background=[('active', '#005a9e')])
        self.estilo.configure('TLabel', background='#f0f0f0', font=('Arial', 12))
        self.estilo.configure('Titulo.TLabel', font=('Arial', 16, 'bold'))

        # Barra de progreso
        self.progress = ttk.Progressbar(master, orient='horizontal', mode='indeterminate')
        self.progress.pack(side='bottom', fill='x')

        # Marco principal
        self.marco = ttk.Frame(master, padding="10 10 10 10")
        self.marco.pack(expand=True, fill='both')

        # Título
        self.titulo = ttk.Label(self.marco, text="Clasificador de Residuos", style='Titulo.TLabel')
        self.titulo.pack(pady=(0, 10))

        # Cargar el modelo en un hilo separado
        threading.Thread(target=self.cargar_modelo, daemon=True).start()

        # Etiqueta para mostrar la imagen
        self.etiqueta_imagen = ttk.Label(self.marco, text="Sube una imagen para clasificar", width=60, anchor='center')
        self.etiqueta_imagen.pack(pady=10)

        # Botones
        self.boton_subir = ttk.Button(self.marco, text="Subir Imagen", command=self.subir_imagen)
        self.boton_subir.pack(pady=5)

        self.boton_subir_varias = ttk.Button(self.marco, text="Subir Múltiples Imágenes", command=self.subir_varias_imagenes)
        self.boton_subir_varias.pack(pady=5)

        # Separador
        self.separador = ttk.Separator(self.marco, orient='horizontal')
        self.separador.pack(fill='x', pady=10)

        # Etiqueta para mostrar resultados
        self.etiqueta_resultado = ttk.Label(self.marco, text="", justify='left')
        self.etiqueta_resultado.pack(pady=10)

        # Historial de clasificaciones
        self.historial_label = ttk.Label(self.marco, text="Historial de Clasificaciones:", font=('Arial', 12, 'bold'))
        self.historial_label.pack(pady=(10, 0))

        self.historial = tk.Text(self.marco, height=8, state='disabled')
        self.historial.pack(pady=5, fill='both', expand=True)

        # Inicializar variables
        self.modelo = None
        self.class_names = ['Battery', 'Biological', 'Cardboard', 'Clothes', 'Glass', 'Metal', 'Paper', 'Plastic', 'Shoes', 'Trash']

    def cargar_modelo(self):
        try:
            self.progress.start()
            self.modelo = load_model('mejor_modelo_garbage.keras')
            self.progress.stop()
            messagebox.showinfo("Información", "Modelo cargado correctamente.")
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"No se pudo cargar el modelo.\n{e}")
            self.master.destroy()

    def subir_imagen(self):
        ruta_archivo = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Archivos de imagen", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if ruta_archivo:
            threading.Thread(target=self.procesar_imagen, args=(ruta_archivo,), daemon=True).start()

    def subir_varias_imagenes(self):
        rutas_archivos = filedialog.askopenfilenames(
            title="Selecciona una o más imágenes",
            filetypes=[("Archivos de imagen", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if rutas_archivos:
            threading.Thread(target=self.procesar_varias_imagenes, args=(rutas_archivos,), daemon=True).start()

    def procesar_imagen(self, ruta_archivo):
        try:
            self.progress.start()
            # Abrir y mostrar la imagen
            imagen = Image.open(ruta_archivo)
            imagen = imagen.resize((300, 300))
            self.imagen_tk = ImageTk.PhotoImage(imagen)
            self.etiqueta_imagen.configure(image=self.imagen_tk, text="")
            self.etiqueta_imagen.image = self.imagen_tk

            # Clasificar la imagen
            resultado = self.clasificar_imagen(ruta_archivo)
            self.mostrar_resultado(resultado)
            self.agregar_a_historial(ruta_archivo, resultado)

            self.progress.stop()

        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"No se pudo procesar la imagen.\n{e}")

    def procesar_varias_imagenes(self, rutas_archivos):
        for ruta in rutas_archivos:
            self.procesar_imagen(ruta)

    def clasificar_imagen(self, ruta):
        # Definir parámetros de la imagen
        altura_imagen, anchura_imagen = 224, 224

        # Cargar y preprocesar la imagen
        img = image.load_img(ruta, target_size=(altura_imagen, anchura_imagen))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Realizar la predicción
        prediccion = self.modelo.predict(img_array)[0]
        porcentaje = prediccion * 100

        # Obtener top 3 predicciones
        top_indices = prediccion.argsort()[-3:][::-1]
        resultados = {self.class_names[i]: porcentaje[i] for i in top_indices}

        return resultados

    def mostrar_resultado(self, resultados):
        texto = "Top 3 predicciones:\n"
        for clase, porcentaje in resultados.items():
            texto += f"{clase}: {porcentaje:.2f}%\n"

        self.etiqueta_resultado.config(text=texto)

    def agregar_a_historial(self, ruta_imagen, resultados):
        nombre_imagen = os.path.basename(ruta_imagen)
        texto = f"{nombre_imagen}:\n"
        for clase, porcentaje in resultados.items():
            texto += f"  {clase}: {porcentaje:.2f}%\n"
        texto += "\n"

        self.historial.configure(state='normal')
        self.historial.insert('end', texto)
        self.historial.configure(state='disabled')

def main():
    root = tk.Tk()
    app = ClasificadorBasuraGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
