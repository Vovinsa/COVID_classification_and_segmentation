from tensorflow.keras.preprocessing.image import img_to_array, load_img
from srv.models import segmenation_model
from os import listdir, mkdir
from os.path import join as joinpath, isdir
from time import time
import argparse

parser = argparse.ArgumentParser(description='Process big pack of images. Put images to process to folder "input" and get it from "output" folder')
parser.add_argument('--mask', dest='make_mask', default=False,
                    help='would put mask images into "output" directory', required=True, type=bool, action=argparse.BooleanOptionalAction)
parser.add_argument('--overlay', dest='make_overlay', default=False,
                    help='would put overlay of image and mask into "output" directory', required=True, type=bool, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

work_dir = "input"
output_dir = "output"
for dir in [work_dir, output_dir]:
    if not isdir(dir):
        print(f"Directory \"{dir}\" is not exist, creating..")
        mkdir(dir)

lung_seg_model = segmenation_model.create_model()
lung_seg_model.load_weights("srv/lungs_segmentation_model_weights/weights.h5")
seg_model = segmenation_model.create_model()
seg_model.load_weights("srv/affections_segmentation_model_weights/weights.h5")
print("Model loaded")

images = listdir(work_dir)
print(f"Loaded {len(images)} images from \"{work_dir}\".")

boot_time = time()
for image in images:
    if image.split(".")[1] not in ["jpg", "png", "jpeg", "webp", "bmp", "gif"]:
        print(f"File {image} incorrect")
        continue
    start_time = time()
    print(f"Working {image}")
    img_orig = img_to_array(load_img(joinpath(work_dir, image)))
    size = img_orig.shape[:2]

    lung_mask = segmenation_model.predict(lung_seg_model, img_orig)
    mask = segmenation_model.predict(seg_model, img_orig)
    if args.make_mask:
        segmenation_model.correct_affections(joinpath(output_dir, "mask_"+image), lung_mask, mask, size)
    if args.make_overlay:
        segmenation_model.visualize_with_mask(
            joinpath(output_dir, "overlay_"+image),
            img_orig, 
            img_to_array(mask[0, ...]), 
            size
            )
    print(f"Image {image} worked in {round(time()-start_time, 2)}s.")
print(f"Done, {len(images)} worked successfully in {round(time()-boot_time, 2)}s.")