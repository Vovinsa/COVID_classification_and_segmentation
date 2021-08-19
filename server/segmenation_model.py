from tensorflow.image import resize
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import cv2


def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def create_model(path):
    model = load_model(path, custom_objects={"iou": iou})
    return model

def predict(model, img):
    img = resize(img, (256, 256))
    mask = model.predict(img[None, ...])
    return img, mask

def visualize(img, mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    img = cv2.addWeighted(img, 1, mask * 255, 0.4, 0.0)
    cv2.imwrite("./detect.png", img)
