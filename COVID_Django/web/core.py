# AI engine
from ntpath import join
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from time import time
from .ai_models import model
from PIL import Image
from os import walk
from os.path import join as join_path, isdir
import zipfile
from shutil import rmtree


class Core:
    def __init__(self):
        super().__init__()
        self.seg_model = model.create_model()
        self.seg_model.load_weights(join_path("web", "weights", "3ch_defeats.h5"))
        self.lung_seg_model = model.create_model()
        self.lung_seg_model.load_weights(join_path("web", "weights", "3ch_lungs.h5"))
    
    def work(self, path, image_stream=None):
        data = {}
        time_start = time()
        img_orig = Image.open(image_stream)
        img_orig = img_orig.convert('RGB')
        img_orig.save(path)
        # img_orig = img_to_array(Image.open(image_stream))
        # img_orig = img_to_array(load_img(path))
        img_orig = img_to_array(img_orig)
        size = img_orig.shape[:2]

        #Segmentation Net
        lungs_output_path = path.split(".")[0] + "_lungs_web.png"
        defeats_output_path = lungs_output_path.replace("lung", "defeat")

            #Mask for lungs
        lung_mask = model.predict(self.lung_seg_model, img_orig)
        lung_pixels = model.visualize(lungs_output_path, lung_mask[0, ...], size)

            #Mask for defeats
        defeat_mask = model.predict(self.seg_model, img_orig)
        defeat_pixels = model.visualize(defeats_output_path, defeat_mask[0, ...], size)

            #Crop defeats
        model.correct_defeats(defeats_output_path, lung_mask, defeat_mask, size)

        data["affections_square"] = round(defeat_pixels/lung_pixels*100)
        data["left_affections"], data["right_affections"] = model.calc_defeats(lungs_output_path, defeats_output_path)
        data["img_url"] = path, lungs_output_path, defeats_output_path
        data["stats"] = {"all_time": round(time() - time_start, 2)}
        return {"success": True, "result": "ok", "data": data}
    

    def work_dicom(self, path, file_stream=None):
        if isdir(path):
            rmtree(path)
        arch = zipfile.ZipFile(file_stream)
        arch.extractall(path)
        arch.close()
        del file_stream
        #TODO NeuralFuck
        result_archive = zipfile.ZipFile(join_path("frontend", "temp", "result.zip"), 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9)
        for root, dirs, files in walk(path):
            for file in files:
                result_archive.write(join_path(root,file), arcname=file)
        return 1
        # data = {}
        # time_start = time()
        # img_orig = Image.open(image_stream)
        # img_orig = img_orig.convert('RGB')
        # img_orig.save(path)
        # # img_orig = img_to_array(Image.open(image_stream))
        # # img_orig = img_to_array(load_img(path))
        # img_orig = img_to_array(img_orig)
        # size = img_orig.shape[:2]

        # #Segmentation Net
        # lungs_output_path = path.split(".")[0] + "_lungs_web.png"
        # defeats_output_path = lungs_output_path.replace("lung", "defeat")

        #     #Mask for lungs
        # lung_mask = model.predict(self.lung_seg_model, img_orig)
        # lung_pixels = model.visualize(lungs_output_path, lung_mask[0, ...], size)

        #     #Mask for defeats
        # defeat_mask = model.predict(self.seg_model, img_orig)
        # defeat_pixels = model.visualize(defeats_output_path, defeat_mask[0, ...], size)

        #     #Crop defeats
        # model.correct_defeats(defeats_output_path, lung_mask, defeat_mask, size)

        # data["affections_square"] = round(defeat_pixels/lung_pixels*100)
        # data["left_affections"], data["right_affections"] = model.calc_defeats(lungs_output_path, defeats_output_path)
        # data["img_url"] = path, lungs_output_path, defeats_output_path
        # data["stats"] = {"all_time": round(time() - time_start, 2)}
        # return {"success": True, "result": "ok", "data": data}