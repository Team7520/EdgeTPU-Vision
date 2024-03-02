import atexit
import collections
import json
import platform
import time

import cv2
from mjpeg_streamer import MjpegServer, Stream
import numpy as np
import tflite_runtime.interpreter as tflite

import argparse

from networktables import NetworkTables
NetworkTables.initialize("localhost")

note_table = NetworkTables.getTable('noteTable')
note_table.putBoolean("Initialized", True)
def exit_handler():
    note_table.putBoolean("Initialized", False)
    time.sleep(0.1)
    # Release the capture when everything is done
    cap.release()
    cv2.destroyAllWindows()


atexit.register(exit_handler)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--weights', required=True, help='Path to the TensorFlow Lite model')
parser.add_argument('--labelmap', required=True, help='Path to the label map file')
parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
args = parser.parse_args()

# Set the model path and label map path
model_path = args.weights
labelmap_path = args.labelmap
min_conf_threshold = args.threshold

# Load the label map into memory
with open(labelmap_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
    print(labels)

delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
    platform.system()
]

# Load the Tensorflow Lite model into memory
interpreter = tflite.Interpreter(model_path,
                          experimental_delegates=[tflite.load_delegate(delegate)])
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

float_input = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

fps_start_time = cv2.getTickCount()
fps_queue = collections.deque(maxlen=30)


# Initialize webcam
cap = cv2.VideoCapture("video.mp4")

# change res

cap.set(3, 1280)  # Set frame width to 1920
cap.set(4, 720)  # Set frame height to 1080

stream = Stream("intakeCam", size=(854, 480), quality=50, fps=30)

server = MjpegServer("localhost", 8080)
server.add_stream(stream)
server.start()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        # Loop video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Resize and process frame
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    imH, imW, _ = frame.shape

    # For NT output
    detections = []

    maxConf = 0
    maxConfObj = {"x": None, "y": None, "w": None, "h": None, "conf": 0}

    # Draw detection results on the frame
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            centerx, centery = (xmin + xmax) / 2, (ymin + ymax) / 2

            detections.append({
                "x": centerx,
                "y": centery,
                "w": xmax - xmin,
                "h": ymax - ymin,
                "conf": scores[i]
            })

            if scores[i] > maxConf:
                maxConf = scores[i]
                maxConfObj = {
                    "x": centerx,
                    "y": centery,
                    "w": xmax - xmin,
                    "h": ymax - ymin,
                    "conf": scores[i]
                }

            # Draw rectangle in openCV

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text



    # Calculate FPS and add it to the queue
    fps_end_time = cv2.getTickCount()
    time_diff = fps_end_time - fps_start_time
    fps = cv2.getTickFrequency() / time_diff
    fps_queue.append(fps)

    # Calculate average FPS over the last 30 frames
    avg_fps = sum(fps_queue) / len(fps_queue)

    fps_text = "Average FPS: {:.2f}".format(avg_fps)
    cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Update the FPS start time for the next frame
    fps_start_time = fps_end_time

    # Send detections to NT

    note_table.putNumber("FPS", avg_fps)
    note_table.putNumber("MaxConf", maxConf)
    note_table.putString("Detections", json.dumps(str(detections)))
    note_table.putString("MaxConfObj", json.dumps(str(maxConfObj)))

    final_frame = frame

    # Display the resulting frame
    # cv2.imshow('Object detector', frame)

    # time.sleep(0.1)
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

    stream.set_frame(frame)

