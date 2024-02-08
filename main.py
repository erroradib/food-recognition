# Adnan Adib
# adnanadib001@gmail.com
import cv2
import numpy as np
import os
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

PATH_TO_FROZEN_GRAPH = 'path/to/frozen_inference_graph.pb'
PATH_TO_LABELS = 'path/to/label_map.pbtxt'

category_index = {}
with open(PATH_TO_LABELS, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'id:' in line:
            id_index = int(line.split(':')[1].strip())
        if 'name:' in line:
            class_name = line.split(':')[1].strip().replace("'", "")
            category_index[id_index] = {'name': class_name}

with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

tf_sess = tf.compat.v1.Session()
tf.import_graph_def(graph_def, name='')

image_tensor = tf_sess.graph.get_tensor_by_name('image_tensor:0')
detection_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
detection_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
detection_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

def detect_objects_tf_gpu(image):
    # Resize image and perform detection on GPU
    image_expanded = np.expand_dims(image, axis=0)
    (boxes, scores, classes, num) = tf_sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    return boxes, scores, classes, num

def detect_objects_tf_cpu(image):
    with tf.device('/CPU:0'):
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = tf_sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

    return boxes, scores, classes, num

dataset_folder = "dataset"

for class_folder in os.listdir(dataset_folder):
    class_folder_path = os.path.join(dataset_folder, class_folder)
    
    if os.path.isdir(class_folder_path):
        print(f"Processing images in folder: {class_folder}")
        
        for image_file in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, image_file)
            
            image = cv2.imread(image_path)
            
            boxes, scores, classes, num = detect_objects_tf_gpu(image)
            
            for i in range(len(boxes[0])):
                if scores[0][i] > 0.5:  
                    class_id = int(classes[0][i])
                    class_name = category_index[class_id]['name']
                    ymin, xmin, ymax, xmax = boxes[0][i]
                    (left, right, top, bottom) = (xmin * image.shape[1], xmax * image.shape[1],
                                                  ymin * image.shape[0], ymax * image.shape[0])
                    cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                    cv2.putText(image, f'{class_name}: {int(scores[0][i]*100)}%', (int(left), int(top - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("Food Recognizer", image)
            cv2.waitKey(0)

cv2.destroyAllWindows()
