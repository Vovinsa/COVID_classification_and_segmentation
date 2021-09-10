# AI engine
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.image import resize
import tensorflow as tf
import json
import numpy as np
from time import time
from srv.models import segmenation_model, detection_model, classification_model

class Core:
    def __init__(self):
        super().__init__()
        
        self.classification_model = classification_model.create_model()
        self.classification_model.load_weights("srv/classification_model_weights/weights.h5")
        self.detect_model = detection_model.create_model(num_classes=2)
        self.detect_model.load_weights("srv/detection_model_weights/weights")
        self.seg_model = segmenation_model.create_model()
        self.seg_model.load_weights("srv/seg_model_w/weights")
    
    def work(self, path):
        time_start = time()
        output_path = path.split(".")[0] + "_worked.png"
        img_orig = img_to_array(load_img(path))

        img, mask = segmenation_model.predict(self.seg_model, img_orig)
        segmenation_model.visualize(output_path, img_to_array(img), img_to_array(mask[0, ...]))

        img = resize(img_orig / 255, (256, 256))
        prediction = self.classification_model.predict(img[None, ...])[0][0]
        result = round(abs(1 - float(prediction)), 2)
        time_first_net = time()
        data = {}
        image = tf.keras.Input(shape=[None, None, 3], name="image")
        predictions = self.detect_model(image, training=False)
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
        if len(boxes) == 0:
            data["left_defeat"], data["right_defeat"] = 0, 0
            data["defeat_sqare"] = 0
            result = 0
        else:
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
            data["left_defeat"], data["right_defeat"] = self.comp_defs(lungs, defeats)
            data["defeat_sqare"] = round(defeats_square / lungs_square * 100)

        data["img_url"] = output_path
        data["stats"] = {"all_time": round(time() - time_start, 2), "first_net": round(time_first_net - time_start, 2), "second_net": round(time() - time_first_net, 2)}
        return json.dumps({"success": True, "result": result, "data": data})
    
    def is_cross(self, a, b):
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

    def comp_defs(self, lungs, defs):
        left_lung, _ = lungs
        lstat = 0
        rstat = 0
        for defeat in defs:
            if self.is_cross(left_lung, defeat):
                lstat += 1
            else:
                rstat += 1
        return lstat, rstat