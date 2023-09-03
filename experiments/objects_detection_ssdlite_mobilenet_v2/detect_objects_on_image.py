import argparse
import numpy as np
import tensorflow as tf

# Definir la versión actual del script
__version__ = '1.0.1'

def detect_objects_on_image(image, model):
    """
    Detecta objetos en una imagen utilizando un modelo de TensorFlow pre-entrenado.

    Args:
        image: np.ndarray, la imagen de entrada como una matriz NumPy.
        model: tf.keras.Model, el modelo de TensorFlow utilizado para la detección de objetos.

    Returns:
        Un diccionario que contiene información sobre los objetos detectados en la imagen.
    """
    # Convertir la imagen a una matriz NumPy si no lo es ya.
    image = np.asarray(image)

    # Convertir la matriz NumPy en un tensor de entrada de TensorFlow.
    input_tensor = tf.convert_to_tensor(image)

    # Agregar una dimensión adicional para que coincida con la forma de entrada del modelo.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Ejecutar el modelo en el tensor de entrada y obtener el diccionario de salida.
    output_dict = model(input_tensor)

    # Obtener el número de detecciones del diccionario de salida y convertirlo a un entero.
    num_detections = int(output_dict['num_detections'])

    # Eliminar la clave 'num_detections' del diccionario de salida y convertir las clases de detección en enteros de 64 bits.
    output_dict = {
        key: value[0, :num_detections].numpy()
        for key, value in output_dict.items()
        if key != 'num_detections'
    }
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict

# Si este archivo se está ejecutando como un programa independiente, imprimir un mensaje de advertencia.
if __name__ == "__main__":
    print("Estás ejecutando solo la función de detección de objetos. Por favor, ejecuta el script principal en su lugar.")
    
    # Definir los argumentos de línea de comandos utilizando argparse.
    parser = argparse.ArgumentParser(description="Detectar objetos en una imagen utilizando un modelo de TensorFlow pre-entrenado.")
    parser.add_argument("image_path", metavar="IMAGE_PATH", type=str, help="La ruta de la imagen de entrada.")
    parser.add_argument("model_path", metavar="MODEL_PATH", type=str, help="La ruta del modelo de TensorFlow.")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    # Imprimir el diccionario de salida.
    print(args)
