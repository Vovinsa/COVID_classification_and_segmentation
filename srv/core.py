# AI engine
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.image import resize
import tensorflow as tf
import json
import numpy as np
from time import time
from srv.models import segmenation_model, classification_model

class Core:
    def __init__(self):
        super().__init__()
        
        self.clf_model = classification_model.create_model()
        self.clf_model.load_weights("srv/classification_model_weights/weights.h5")
        self.seg_model = segmenation_model.create_model()
        self.seg_model.load_weights("srv/segmentation_model_weights/weights")
        self.lung_seg_model = segmenation_model.create_model()
        self.lung_seg_model.load_weights("srv/lungs_segmentation_model_weights/weights.h5")
    
    def work(self, path):
        data = {}
        time_start = time()
        img_orig = img_to_array(load_img(path))
        size = img_orig.shape[:2]
        
        #Classification Net
        img = resize(img_orig / 255, (256, 256))
        prediction = self.clf_model.predict(img[None, ...])[0][0]
        result = round(abs(1 - float(prediction)), 2)
        time_first_net = time()

        #Segmentation Net
        lungs_output_path = path.split(".")[0] + "_lungs.png"
        defeats_output_path = lungs_output_path.replace("lung", "defeat")

        #Mask for lungs
        img, mask = segmenation_model.predict(self.lung_seg_model, img_orig)
        lung_pixels = segmenation_model.visualize(lungs_output_path, img_to_array(mask[0, ...]), size)

        #Mask for defeats
        img, mask = segmenation_model.predict(self.seg_model, img_orig)
        defeat_pixels = segmenation_model.visualize(defeats_output_path, img_to_array(mask[0, ...]), size)

        data["defeat_square"] = round(defeat_pixels / lung_pixels * 100)
        data["left_defeat"], data["right_defeat"] = segmenation_model.calc_defeats(lungs_output_path, defeats_output_path)
        data["img_url"] = path, lungs_output_path, defeats_output_path
        data["stats"] = {"all_time": round(time() - time_start, 2), "first_net": round(time_first_net - time_start, 2), "second_net": round(time() - time_first_net, 2)}
        return json.dumps({"success": True, "result": result, "data": data})
