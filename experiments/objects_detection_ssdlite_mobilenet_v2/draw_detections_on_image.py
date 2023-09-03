import argparse
import cv2
import math


# Definir la versión actual del script
__version__ = '1.0.1'


def draw_detections_on_image(image, detections, labels):
    """
    Dibuja los cuadros de detección y las etiquetas en la imagen.
    
    Args:
    - image: imagen de entrada.
    - detections: diccionario con las detecciones obtenidas de un modelo de detección de objetos.
    - labels: lista de etiquetas de los objetos que el modelo puede detectar.
    
    Returns:
    - image_with_detections: imagen con los cuadros de detección y las etiquetas dibujadas.
    """
    # Copiar la imagen para no modificar la original.
    image_with_detections = image.copy()
    
    # Obtener las dimensiones de la imagen.
    height, width, channels = image_with_detections.shape
    
    # Configurar el texto de las etiquetas.
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)
    label_padding = 5
    label_text_size = 1.4 # tamaño de la etiqueta que describe la clase del objeto
    
    # Obtener el número de detecciones y procesar cada una.
    num_detections = detections['num_detections']
    if num_detections > 0:
        for detection_index in range(num_detections):
            # Obtener la puntuación, la caja delimitadora y la clase de la detección.
            detection_score = detections['detection_scores'][detection_index]
            detection_box = detections['detection_boxes'][detection_index]
            detection_class = detections['detection_classes'][detection_index]
            detection_label = labels[detection_class]
            detection_label_full = detection_label + ' ' + str(math.floor(100 * detection_score)) + '%'
            
            # Calcular las coordenadas de la caja de detección.
            y1 = int(height * detection_box[0])
            x1 = int(width * detection_box[1])
            y2 = int(height * detection_box[2])
            x2 = int(width * detection_box[3])
                        
            # Dibujar el rectángulo de la detección.
            image_with_detections = cv2.rectangle(
                image_with_detections,
                (x1, y1),
                (x2, y2),
                color,
                3
            )
            
            # Dibujar el fondo de la etiqueta.
            label_size = cv2.getTextSize(
                detection_label_full,
                cv2.FONT_HERSHEY_COMPLEX,
                label_text_size,
                2
            )
            image_with_detections = cv2.rectangle(
                image_with_detections,
                (x1, y1 - label_size[0][1] - 2 * label_padding),
                (x1 + label_size[0][0] + 2 * label_padding, y1),
                color,
                -1
            )
            
            # Dibujar el texto de la etiqueta.
            cv2.putText(
                image_with_detections,
                detection_label_full,
                (x1 + label_padding, y1 - label_padding),
                font,
                label_text_size,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
            
    return image_with_detections

if __name__ == '__main__':
    print("Estás ejecutando solo la función de dibujar sobre los objetos detectados. Por favor, ejecuta el script principal en su lugar.")

    parser = argparse.ArgumentParser(description='Draw object detection results on an image')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
    args = parser.parse_args()
    
    print(args)
