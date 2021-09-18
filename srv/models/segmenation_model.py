from tensorflow.image import resize
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Input, Concatenate, MaxPool2D, UpSampling2D, Add
from tensorflow.keras.models import Model
import cv2
import numpy as np
from PIL import Image
from skimage import morphology, measure
from sklearn.cluster import KMeans

def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
    union = K.sum(y_true, [1, 2]) + K.sum(y_pred, [1, 2]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def resblock(X, f):
    X_copy = X
    
    X = Conv2D(f, kernel_size=(1, 1), kernel_initializer="he_normal")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    
    X = Conv2D(f, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal")(X)
    X = BatchNormalization()(X)
    
    X_copy = Conv2D(f, kernel_size=(1, 1), kernel_initializer="he_normal")(X_copy)
    X_copy = BatchNormalization()(X_copy)
    
    X = Add()([X, X_copy])
    X = Activation("relu")(X)
    
    return X

def upsample_concat(x, skip):
    X = UpSampling2D((2, 2))(x)
    merge = Concatenate()([X, skip])
    
    return merge

def create_model():
    input_shape = (256, 256, 3)
    X_input = Input(input_shape)

    conv_1 = Conv2D(16, 3, activation="relu", padding="same", kernel_initializer="he_normal")(X_input)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Conv2D(16, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    pool_1 = MaxPool2D((2,2))(conv_1)

    conv_2 = resblock(pool_1, 32)
    pool_2 = MaxPool2D((2,2))(conv_2)

    conv_3 = resblock(pool_2, 64)
    pool_3 = MaxPool2D((2,2))(conv_3)

    conv_4 = resblock(pool_3, 128)
    pool_4 = MaxPool2D((2,2))(conv_4)

    conv_5 = resblock(pool_4, 256)

    up_1 = upsample_concat(conv_5, conv_4)
    up_1 = resblock(up_1, 128)

    up_2 = upsample_concat(up_1, conv_3)    
    up_2 = resblock(up_2, 64)

    up_3 = upsample_concat(up_2, conv_2)
    up_3 = resblock(up_3, 32)

    up_4 = upsample_concat(up_3, conv_1)
    up_4 = resblock(up_4, 16)

    output = Conv2D(3, (1,1), kernel_initializer="he_normal", padding="same", activation="sigmoid")(up_4)

    model = Model(X_input, output)

    return model

def predict(model, img):
    img = resize(img, (256, 256))
    mask = model.predict(img[None, ...])
    return img, mask


def visualize(path, mask, size=None):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    if size:
        mask = np.array(resize(mask, size))
    cv2.imwrite(path, mask * 255)
    mask = np.around(mask)
    cv2.imwrite(path.replace("_web", ""), mask * 255)

    return np.sum(mask)

def visualize_with_mask(path, img, mask, size=None):
    mask *= 255
    if size:
        mask = np.array(resize(mask, size))
    img = cv2.addWeighted(img, 1, mask, 0.4, 0.0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img)
    return


def calc_defeats(lungs_output_path, defeats_output_path):
    src = np.array(Image.open(defeats_output_path.replace("_web", "")))
    mask_orig = Image.open(lungs_output_path.replace("_web", ""))
    mask = np.array(mask_orig.resize(src.shape[1::-1], Image.BILINEAR))

    mask[:, :, 1] = mask[:, :, 0]
    mask[:, :, 2] = mask[:, :, 0]
    mask = mask / 255
    ldst = src * mask
    left = count_points(np.around(ldst/255)*255)
    
    mask = np.array(mask_orig.resize(src.shape[1::-1], Image.BILINEAR))
    mask[:, :, 0] = mask[:, :, 1]
    mask[:, :, 2] = mask[:, :, 1]
    mask = mask / 255
    rdst = src * mask
    right = count_points(np.around(rdst / 255) * 255)
    return left, right


def count_points(img):
    rows, cols, bands = img.shape
    X = img.reshape(rows * cols, bands)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    labels = kmeans.labels_.reshape(rows, cols)
    all_count=0
    for i in np.unique(labels):
        blobs = np.int_(morphology.binary_opening(labels == i))
        color = list(np.around(kmeans.cluster_centers_[i]))
        if color==[0,0,0]:
            continue
        count = len(np.unique(measure.label(blobs))) - 1
        all_count += count
    return all_count
