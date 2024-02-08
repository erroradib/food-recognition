import cv2
import numpy as np
import os
import tensorflow as tf

# Set GPU memory growth to avoid allocating all memory at once
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Load the TensorFlow frozen inference graph and corresponding labels
PATH_TO_FROZEN_GRAPH = 'path/to/frozen_inference_graph.pb'
PATH_TO_LABELS = 'path/to/label_map.pbtxt'

# Load label map
category_index = {}
with open(PATH_TO_LABELS, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'id:' in line:
            id_index = int(line.split(':')[1].strip())
        if 'name:' in line:
            class_name = line.split(':')[1].strip().replace("'", "")
            category_index[id_index] = {'name': class_name}

# Load TensorFlow frozen graph for detection
with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# Create TensorFlow session and load the graph
tf_sess = tf.compat.v1.Session()
tf.import_graph_def(graph_def, name='')

# Get input and output tensors
image_tensor = tf_sess.graph.get_tensor_by_name('image_tensor:0')
detection_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
detection_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
detection_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

# Function to detect objects in an image using TensorFlow with GPU
def detect_objects_tf_gpu(image):
    # Resize image and perform detection on GPU
    image_expanded = np.expand_dims(image, axis=0)
    (boxes, scores, classes, num) = tf_sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    return boxes, scores, classes, num

# Function to detect objects in an image using TensorFlow with CPU
def detect_objects_tf_cpu(image):
    with tf.device('/CPU:0'):
        # Resize image and perform detection on CPU
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = tf_sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

    return boxes, scores, classes, num

# Path to the dataset folder
dataset_folder = "dataset"

# Loop over the subfolders in the dataset folder
for class_folder in os.listdir(dataset_folder):
    class_folder_path = os.path.join(dataset_folder, class_folder)
    
    # Ensure it's a directory
    if os.path.isdir(class_folder_path):
        print(f"Processing images in folder: {class_folder}")
        
        # Loop over the images in the class folder
        for image_file in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, image_file)
            
            # Read the image
            image = cv2.imread(image_path)
            
            # Detect objects in the image using TensorFlow with GPU
            boxes, scores, classes, num = detect_objects_tf_gpu(image)
            
            # Draw bounding boxes on the image
            for i in range(len(boxes[0])):
                if scores[0][i] > 0.5:  # Adjust confidence threshold as needed
                    class_id = int(classes[0][i])
                    class_name = category_index[class_id]['name']
                    ymin, xmin, ymax, xmax = boxes[0][i]
                    (left, right, top, bottom) = (xmin * image.shape[1], xmax * image.shape[1],
                                                  ymin * image.shape[0], ymax * image.shape[0])
                    cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                    cv2.putText(image, f'{class_name}: {int(scores[0][i]*100)}%', (int(left), int(top - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display the image with detected objects
            cv2.imshow("Food Recognizer", image)
            cv2.waitKey(0)

# Close OpenCV window
cv2.destroyAllWindows()
