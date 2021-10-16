# AI engine
from tensorflow.image import resize
from tensorflow.keras.preprocessing.image import img_to_array
from time import time
from .ai_models import model
from PIL import Image
from os.path import join as join_path, isdir
import zipfile
from shutil import rmtree
import os
import pydicom as dicom
import SimpleITK as sitk
import numpy as np
import cv2

class Core:
    def __init__(self):
        super().__init__()
        self.defeats_3ch = model.create_model(3, 3)
        self.defeats_3ch.load_weights(join_path("web", "weights", "3ch_defeats.h5"))
        self.lungs_3ch = model.create_model(3, 3)
        self.lungs_3ch.load_weights(join_path("web", "weights", "3ch_lungs.h5"))

        self.defeats_1ch = model.create_model(1, 1)
        self.defeats_1ch.load_weights(join_path("web", "weights", "1ch_defeats_mosmed.h5"))
        self.lungs_1ch = model.create_model(1, 2)
        self.lungs_1ch.load_weights(join_path("web", "weights", "1ch_lungs_kristi.h5"))
    
    def work(self, path, image_stream=None):
        data = {}
        time_start = time()
        img_orig = Image.open(image_stream)
        img_orig = img_orig.convert('RGB')
        img_orig.save(path)
        img_orig = img_to_array(img_orig)
        size = img_orig.shape[:2]

        #Segmentation Net
        lungs_output_path = path.split(".")[0] + "_lungs_web.png"
        defeats_output_path = lungs_output_path.replace("lung", "defeat")

        #Mask for lungs
        lung_mask = model.predict(self.lungs_3ch, img_orig)
        lung_pixels = model.visualize(lungs_output_path, lung_mask[0, ...], size)

        #Mask for defeats
        defeat_mask = model.predict(self.defeats_3ch, img_orig)
        defeat_pixels = model.visualize(defeats_output_path, defeat_mask[0, ...], size)

        #Crop defeats
        model.correct_defeats(defeats_output_path, lung_mask, defeat_mask, size)

        data["affections_square"] = round(defeat_pixels/lung_pixels*100)
        data["left_affections"], data["right_affections"] = model.calc_defeats(lungs_output_path, defeats_output_path)
        data["img_url"] = path, lungs_output_path, defeats_output_path
        data["stats"] = {"all_time": round(time() - time_start, 2)}
        return {"success": True, "result": "ok", "is_dicom": False, "data": data}

    def work_dicom(self, path, file_stream):
        if isdir(path):
            rmtree(path)
        arch = zipfile.ZipFile(file_stream)
        arch.extractall(path)
        arch.close()
        del file_stream

        time_start = time()

        # Dicom model works
        slices = os.listdir(path)
        left_defeats_volume = 0
        right_defeats_volume = 0

        info_img = dicom.read_file(join_path(path, slices[0]))
        spacing_between_slices = 3
        slice_thickness =  info_img.SliceThickness
        slice_distance = slice_thickness + spacing_between_slices
        pixel_size = np.mean(info_img.PixelSpacing)
        image_size = info_img.pixel_array[..., None].shape

        jpg_paths_orig = []
        jpg_paths_overlay = []

        left_affection_percent = []
        right_affection_percent = []
        i=0
        worked_slices = []
        
        result_dir_path = join_path(path, "result")
        if not isdir(result_dir_path):
            os.mkdir(result_dir_path)
        
        for slice_name in enumerate(slices):
            if slice_name[1] == "__MACOSX":
                continue
            slice_path = join_path(path, slice_name[1])
            file = dicom.read_file(slice_path)
            img = file.pixel_array[..., None]
            
            mask_affections = model.predict_dicom(self.defeats_1ch, img)
            mask_affections = np.around(mask_affections)

            mask_lungs = model.predict_dicom(self.lungs_1ch, img)
            mask_lungs = np.around(mask_lungs)

            overlay = img[..., 0] + resize(mask_affections[0, ...], (image_size[0], image_size[1]))[..., 0]
            overlay = np.array(overlay)
            worked_slices.append(overlay)

            left_lung = mask_lungs[..., 0]
            right_lung = mask_lungs[..., 1]

            #Count of lung pixels
            left_lung_pixels = np.sum(left_lung)
            right_lung_pixels = np.sum(right_lung)

            #Count affection square in pixels
            left_defeats_pixels = np.sum(left_lung*mask_affections)
            right_defeats_pixels = np.sum(right_lung*mask_affections)

            #Calc affection percent
            left_affection_percent.append(left_defeats_pixels / left_lung_pixels)
            right_affection_percent.append(right_defeats_pixels / right_lung_pixels)

            #Calc affection square in mm2
            left_defeats_mm2 = left_defeats_pixels * pixel_size 
            right_defeats_mm2 = right_defeats_pixels * pixel_size 

            #Calc affection square in mm3
            left_defeats_volume += left_defeats_mm2 * slice_distance 
            right_defeats_volume += right_defeats_mm2 * slice_distance

            if i == 2:
                i = 0
                orig_path = join_path(result_dir_path, slice_name[1] + ".jpg")
                overlay_path = join_path(result_dir_path, slice_name[1] + "_overlay.jpg")
                cv2.imwrite(orig_path, file.pixel_array)
                jpg_paths_orig.append(orig_path)
                cv2.imwrite(overlay_path, overlay)
                jpg_paths_overlay.append(overlay_path)
            i += 1
   
        left_affection_percent = np.mean(left_affection_percent)
        right_affection_percent = np.mean(right_affection_percent)
        
        pred_slice = np.array(worked_slices, dtype=np.uint16)
        pred_slice = sitk.GetImageFromArray(pred_slice)
        result_path = join_path(path, "result", f"{slice_name[0]}.dcm")
        sitk.WriteImage(pred_slice, result_path)
    
        data = {}
        data["left_affection_percent"], data["right_affection_percent"] = round(left_affection_percent * 100, 2), round(right_affection_percent * 100, 2)
        data["left_defeats_volume"], data["right_defeats_volume"] = round(left_defeats_volume / 1000, 2), round(right_defeats_volume / 1000, 2)
        data["img_urls"] = jpg_paths_orig
        data["archive"] = result_path
        data["slice_count"] = len(slices)
        data["stats"] = {"all_time": round(time() - time_start, 2)}
        return {"success": True, "result": "ok", "data": data}
