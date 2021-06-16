import pytesseract
from imutils.object_detection import non_max_suppression
import cv2
import numpy as np

min_confidence = 0.5

def decode_predictions(scores, geometry):   
    (num_rows, num_cols) = scores.shape[2:4]
    rects, confidences = [], []
    for y in range(0, num_rows):
        scores_data = scores[0, 0, y]
        x_data0, x_data1 = geometry[0, 0, y], geometry[0, 1, y]
        x_data2, x_data3 = geometry[0, 2, y], geometry[0, 3, y]
        angles_data = geometry[0, 4, y]
        for x in range(0, num_cols):
            if scores_data[x] < min_confidence: continue 
            (offset_x, offset_y) = (x * 4.0, y * 4.0) 
            angle = angles_data[x]
            cos, sin = np.cos(angle), np.sin(angle) 
            h, w = x_data0[x] + x_data2[x], x_data1[x] + x_data3[x] 
            end_x = int(offset_x + (cos * x_data1[x]) + \
                    (sin * x_data2[x]))
            end_y = int(offset_y - (sin * x_data1[x]) + \
                    (cos * x_data2[x]))
            start_x, start_y = int(end_x - w), int(end_y - h) 
            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x]) 
    return (rects, confidences)

file = './picture.png'
image = cv2.imread(file)
orig = image.copy()
(orig_height, orig_width) = image.shape[:2]
width = height = 32*10
(w, h) = (width, height)
r_width, r_height = orig_width / float(w), orig_height / float(h)
image = cv2.resize(image, (w, h))
(h, w) = image.shape[:2]

layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
net = cv2.dnn.readNet('models/text_detection/frozen_east_text_detection.pb')

b, g, r = np.mean(image[...,0]), np.mean(image[...,1]), np.mean(image[...,2])
blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (b, g, r), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layer_names) 


(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

padding = 0.001 #0.01 #0.5
results = []
# loop over the bounding boxes
for (start_x, start_y, end_x, end_y) in boxes:
    start_x, start_y = int(start_x*r_width), int(start_y*r_height)
    end_x, end_y = int(end_x*r_width), int(end_y*r_height)
    d_x, d_y = int((end_x - start_x) * padding), \
               int((end_y - start_y) * padding)
    start_x, start_y = max(0, start_x - d_x*2), \
                       max(0, start_y - d_y*2)
    end_x, end_y = min(orig_width, end_x + (d_x * 2)), \
                   min(orig_height, end_y + (d_y * 2))
    roi = orig[start_y:end_y, start_x:end_x]
    config = ("-l fra --oem 1 --psm 11")
    text = pytesseract.image_to_string(roi, config=config)
    results.append(((start_x, start_y, end_x, end_y), text))
    results = sorted(results, key=lambda r:r[0][1])

output = orig.copy()
for ((start_x, start_y, end_x, end_y), text) in results:
    # strip out non-ASCII text so we can draw the text on the image 
    text = "".join([c if ord(c) < 128 else "" for c \
            in text]).strip()
    cv2.rectangle(output, (start_x, start_y), (end_x, end_y), \
                          (0, 255, 0), 2)
    cv2.putText(output, text, (start_x, start_y-20), \
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)    

print(text)
cv2.imwrite('./result.jpg', output)
