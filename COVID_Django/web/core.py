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
import matplotlib.pyplot as plt
import cv2
import nibabel as nib


class Core:
    def __init__(self):
        super().__init__()
        self.defeats_3ch = model.create_model(3, 3)
        self.defeats_3ch.load_weights(join_path("web", "weights", "3ch_defeats.h5"))
        self.lungs_3ch = model.create_model(3, 3)
        self.lungs_3ch.load_weights(join_path("web", "weights", "3ch_lungs.h5"))

        self.defeats_1ch = model.create_model(1, 1)
        self.defeats_1ch.load_weights(join_path("web", "weights", "1ch_defeats.h5"))
        self.lungs_1ch = model.create_model(1, 2)
        self.lungs_1ch.load_weights(join_path("web", "weights", "1ch_lungs.h5"))
    
    def work(self, path, image_stream=None):
        data = {}
        time_start = time()
        img_orig = Image.open(image_stream)
        img_orig = img_orig.convert("RGB")
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
        # Extract incoming zip
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

        is_nibabel = False
        if slices[0].split(".")[-1] in ["nii", "gz"]:
            is_nibabel = True
        if is_nibabel:
            spacing_between_slices = 3
            slice_thickness = 3
            pixel_size = 0.7
        else:
            info_img = dicom.read_file(join_path(path, slices[0]))
            spacing_between_slices = 0
            slice_thickness = info_img.SliceThickness
            pixel_size = np.mean(info_img.PixelSpacing)
            image_size = info_img.pixel_array[..., None].shape
        slice_distance = slice_thickness + spacing_between_slices

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
            if is_nibabel:
                file = nib.load(slice_path)
                img = file.get_fdata()
                image_size = img.shape
            else:
                file = dicom.read_file(slice_path)
                img = file.pixel_array[..., None]
            
            mask_affections = model.predict_dicom(self.defeats_1ch, img)
            mask_affections_round = np.round(mask_affections)

            mask_lungs = model.predict_dicom(self.lungs_1ch, img)
            mask_lungs = np.round(mask_lungs)

            overlay = (img / 2048 + 1 + resize(mask_affections_round[0, ...], (image_size[0], image_size[1]))) * 255
            overlay = np.array(overlay[..., 0])
            worked_slices.append(overlay)

            left_lung = mask_lungs[..., 0]
            right_lung = mask_lungs[..., 1]

            #Count of lung pixels
            left_lung_pixels = np.sum(left_lung)
            right_lung_pixels = np.sum(right_lung)

            #Crop affections by mask
            left_affection_crop = np.sum(left_lung * mask_affections_round)
            right_affection_crop = np.sum(right_lung * mask_affections_round)

            #Count affection square in pixels
            left_defeats_pixels = np.sum(left_affection_crop)
            right_defeats_pixels = np.sum(right_affection_crop)

            #Calc affection percent
            if left_defeats_pixels > 0:
                left_affection_percent.append(left_defeats_pixels / left_lung_pixels)
            if right_defeats_pixels > 0:
                right_affection_percent.append(right_defeats_pixels / right_lung_pixels)

            #Calc affection square in mmm2
            left_defeats_mm2 = left_affection_crop * pixel_size 
            right_defeats_mm2 = right_affection_crop * pixel_size 

            #Calc affection square in mm3
            left_defeats_volume += left_defeats_mm2 * slice_distance 
            right_defeats_volume += right_defeats_mm2 * slice_distance

            if i==2:
                i = 0
                orig_path = join_path(result_dir_path, slice_name[1]+".png")
                jpg_paths_orig.append(orig_path)
                overlay_path = join_path(result_dir_path, slice_name[1]+"_overlay.png")
                jpg_paths_overlay.append(overlay_path)
                cv2.imwrite(orig_path, (img / 2048 + 1) * 100)
                overlay = np.zeros((image_size[0], image_size[1], 4), np.uint8)
                overlay[:, :, 2] = overlay[:, :, 3] = (resize(mask_affections[0, ...], (image_size[0], image_size[1])) * 255)[..., 0]
                cv2.imwrite(overlay_path, np.around(overlay))
            i+=1

        left_affection_percent_result = np.sum(left_affection_percent) / len(slices)
        right_affection_percent_result = np.sum(right_affection_percent) / len(slices)

        pred_slice = np.array(worked_slices, dtype=np.uint16)
        print(pred_slice.shape)
        pred_slice = sitk.GetImageFromArray(pred_slice)
        if is_nibabel:
            result_path = join_path(path, "result", f"{slice_name[0]}.dcm")
        else:
            result_path = join_path(path, "result", f"{slice_name[0]}.dcm")
        
        sitk.WriteImage(pred_slice, result_path)
    
        data = {}
        data["left_affection_percent"], data["right_affection_percent"] = np.round(left_affection_percent_result * 100, 2), np.round(right_affection_percent_result * 100, 2)
        data["left_defeats_volume"], data["right_defeats_volume"] = np.round(left_defeats_volume / 10000, 2), np.round(right_defeats_volume / 1000, 2)
        data["img_urls"] = jpg_paths_orig, jpg_paths_overlay
        data["archive"] = result_path
        data["slice_count"] = len(slices)
        data["stats"] = {"all_time": round(time() - time_start, 2)}
        return {"success": True, "result": "ok", "data": data}