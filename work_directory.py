from tensorflow.keras.preprocessing.image import img_to_array, load_img
from srv.models import segmenation_model
from os import listdir, mkdir
from os.path import join as joinpath, isdir
from time import time

work_dir = "input"
output_dir = "output"
for dir in [work_dir, output_dir]:
    if not isdir(dir):
        print(f"Directory \"{dir}\" is not exist, creating..")
        mkdir(dir)
seg_model = segmenation_model.create_model()
seg_model.load_weights("srv/seg_model_w/weights")
print("Model loaded")

images = listdir(work_dir)
print(f"Loaded {len(images)} images from \"{work_dir}\".")
boot_time = time()
for image in images:
    if image.split(".")[1] not in ["jpg", "png", "jpeg", "webp", "bmp"]:
        print(f"File {image} incorrect")
        continue
    start_time = time()
    print(f"Working {image}")
    img_orig = img_to_array(load_img(joinpath(work_dir, image)))
    img, mask = segmenation_model.predict(seg_model, img_orig)
    segmenation_model.visualize(joinpath(output_dir, image), img_to_array(img), img_to_array(mask[0, ...]))
    print(f"Image {image} worked in {round(time()-start_time, 2)} sec")
print(f"Done, {len(images)} worked successfully in {round(time()-boot_time, 2)} sec.")