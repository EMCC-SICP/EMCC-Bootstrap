import cv2
import object_detection
import draw_detections_on_image
import detect_objects_on_image
import argparse

# Definir la versión actual del script
__version__ = '1.0.1'

if __name__ == '__main__':
    # Definir los argumentos de línea de comandos utilizando argparse.
    parser = argparse.ArgumentParser(description='Detección de objetos en un video en tiempo real.')
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}', help='Muestra la versión del programa.')
    args = parser.parse_args()

    #Diferentes modelos para hacer pruebas
    centernet = './Modelos/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8/saved_model' #17.6s
    efficiendet_d1 = './Modelos/efficientdet_d1_coco17_tpu-32/saved_model'              #24.6s
    ssd = './Modelos/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model'                 #9.9s
    #Primer modelo que se usó
    ssdlite = 'experiments/objects_detection_ssdlite_mobilenet_v2/Modelos/ssdlite_mobilenet_v2_coco_2018_05_09/saved_model'              #4.9s


    MODEL_DOWNLOADED = ssdlite

    saved_model = object_detection.load_model_downloaded(MODEL_DOWNLOADED)
    # Loading default model signature.
    model = saved_model.signatures['serving_default']

    LABELS_NAME = 'experiments/objects_detection_ssdlite_mobilenet_v2/Label_maps/mscoco_label_map.pbtxt'
    labels = object_detection.load_labels_downloaded(LABELS_NAME)

    # Definir un objeto de captura de video
    vid = cv2.VideoCapture(0)

    while True:
        # Capturar el fotograma del video
        ret, frame = vid.read()

        # Realizar la detección de objetos en la imagen
        detections = detect_objects_on_image.detect_objects_on_image(frame, model)

        # Dibujar las detecciones en la imagen
        image_with_detections = draw_detections_on_image.draw_detections_on_image(frame, detections, labels)

        # Mostrar el marco con las detecciones
        cv2.imshow('frame', image_with_detections)

        # La tecla 'q' se configura como la tecla para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Después del bucle, liberar el objeto de captura
    vid.release()

    # Cerrar todas las ventanas
    cv2.destroyAllWindows()
