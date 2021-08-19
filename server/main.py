from flask import Flask, request, Response
import classification_model
import detection_model 
import segmenation_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.image import resize
import tensorflow as tf
import json
import numpy as np
from time import time

app = Flask(__name__)

def is_cross(a, b):
    xA = [a[0], a[2]]
    xB = [b[0], b[2]]

    yA = [a[1], a[3]]
    yB = [b[1], b[3]]

    if max(xA) < min(xB) or max(yA) < min(yB) or min(yA) > max(yB):
        return False
    elif max(xA) > min(xB) and min(xA) < min(xB):
        return True
    else:
        return True

def comp_defs(lungs, defs):
    left_lung, _ = lungs
    lstat = 0
    rstat = 0
    for defeat in defs:
        if is_cross(left_lung, defeat):
            lstat += 1
        else:
            rstat += 1
    return lstat, rstat

classification_model = classification_model.create_model()
classification_model.load_weights("server/classification_model_weights/weights.h5")

detect_model = detection_model.create_model(2)
detect_model.load_weights("server/detection_model_weights/weights")

seg_model = segmenation_model.create_model("server/seg_model")

@app.route("/predict", methods=["POST"])
def predict_covid():
    imagefile = request.files["image"]
    imagefile.save("./detect.png")
    time_start = time()

    img_orig = img_to_array(load_img("./detect.png"))

    img, mask = segmenation_model.predict(seg_model, img_orig)
    segmenation_model.visualize(img_to_array(img), img_to_array(mask[0, ...]))

    img = resize(img_orig / 255, (256, 256))
    prediction = classification_model.predict(img[None, ...])[0][0]
    result = round(abs(1 - float(prediction)), 2)
    time_first_net = time()
    data = {}
    image = tf.keras.Input(shape=[None, None, 3], name="image")
    predictions = detect_model(image, training=False)
    detections = detection_model.DecodePredictions(confidence_threshold=0.75)(image, predictions)
    inference_model = tf.keras.Model(inputs=image, outputs=detections)

    image = tf.cast(img_orig, dtype=tf.float32)
    input_image, ratio = detection_model.prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = []

    for x in detections.nmsed_classes[0][:num_detections]:
        if x == 0:
            class_names.append("Lung")
        else:
            class_names.append("Defeat")
    scores = np.array(detections.nmsed_scores[0][:num_detections]).tolist()
    boxes = np.array(detections.nmsed_boxes[0][:num_detections] / ratio).tolist()
    data["detections"] = []
    defeats_square = 0
    lungs_square = 0
    defeats = []
    lungs = []
    for index, box in enumerate(boxes):
        box = boxes[index]
        class_name = class_names[index]
        x1, y1, x2, y2 = box
        square = (x2 - x1) * (y2 - y1)
        if class_name == "Defeat":
            defeats_square += square
            defeats.append(box)
        else:
            lungs_square += square
            lungs.append(box)
        data["detections"].append({"score": round(scores[index], 2), "box": box, "class_name": class_name})
    lungs.sort(key = lambda x: x[0])
    data["left_defeat"], data["right_defeat"] = comp_defs(lungs, defeats)
    data["defeat_sqare"] = round(defeats_square / lungs_square * 100)
    data["img_url"] = "./detect.png"
    
    data["stats"] = {"all_time": round(time() - time_start, 2), "first_net": round(time_first_net - time_start, 2), "second_net": round(time() - time_first_net, 2)}
    response = json.dumps({"success": True, "result": result, "data": data})
    resp = Response(response)
    resp.headers["Access-Control-Allow-Origin"] = '*'

    return resp

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)