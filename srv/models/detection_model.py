from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow import keras
import tensorflow as tf
import numpy as np

def convert_to_xywh(boxes):
  return tf.concat([(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]], axis=-1)

def convert_to_corners(boxes):
  return tf.concat([boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0], axis=-1)

class AnchorBox:
  def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

  def _compute_dims(self):
    anchor_dims_all = []
    for area in self._areas:
      anchor_dims = []
      for ratio in self.aspect_ratios:
        anchor_height = tf.math.sqrt(area / ratio)
        anchor_width = area / anchor_height
        dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
        for scale in self.scales:
          anchor_dims.append(scale * dims)
      anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
    return anchor_dims_all

  def _get_anchors(self, feature_height, feature_width, level):
    rx = tf.range(feature_width, dtype=tf.float32) + 0.5
    ry = tf.range(feature_height, dtype=tf.float32) + 0.5
    centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
    centers = tf.expand_dims(centers, axis=-2)
    centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
    dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
    anchors = tf.concat([centers, dims], axis=-1)
    return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

  def get_anchors(self, image_height, image_width):
    anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2 ** i),
                tf.math.ceil(image_width / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
    return tf.concat(anchors, axis=0)

def random_flip_horizontal(image, boxes):
  if tf.random.uniform(()) > 0.5:
    image = tf.image.flip_left_right(image)
    boxes = tf.stack([1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1)

  return image, boxes


def resize_and_pad_image(image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0):
  image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
  if jitter is not None:
    min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
  ratio = min_side / tf.reduce_min(image_shape)
  if ratio * tf.reduce_max(image_shape) > max_side:
      ratio = max_side / tf.reduce_max(image_shape)
  image_shape = ratio * image_shape
  image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
  padded_image_shape = tf.cast(tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32)
  image = tf.image.pad_to_bounding_box(image, 0, 0, padded_image_shape[0], padded_image_shape[1])
  
  return image, image_shape, ratio

class DecodePredictions(layers.Layer):
  def __init__(
        self,
        num_classes=80,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

  def _decode_box_predictions(self, anchor_boxes, box_predictions):
    boxes = box_predictions * self._box_variance
    boxes = tf.concat(
        [
            boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
            tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
        ],
        axis=-1,
    )
    boxes_transformed = convert_to_corners(boxes)
    
    return boxes_transformed

  def call(self, images, predictions):
    image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
    anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
    box_predictions = predictions[:, :, :4]
    cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
    boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)

    return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )

def get_backbone():
  backbone = ResNet50(False, "imagenet", input_shape=[None, None, 3])
  c3_output, c4_output, c5_output = [backbone.get_layer(layer_name).output for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]]
  
  return keras.Model(inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output])

class FeaturePyramid(layers.Layer):
  def __init__(self, backbone=None, **kwargs):
    super(FeaturePyramid, self).__init__(name="FeaturePiramid", **kwargs)
    self.backbone = backbone if backbone else get_backbone()
    self.conv_c3_1x1 = layers.Conv2D(256, 1, 1, "same")
    self.conv_c4_1x1 = layers.Conv2D(256, 1, 1, "same")
    self.conv_c5_1x1 = layers.Conv2D(256, 1, 1, "same")
    self.conv_c3_3x3 = layers.Conv2D(256, 3, 1, "same")
    self.conv_c4_3x3 = layers.Conv2D(256, 3, 1, "same")
    self.conv_c5_3x3 = layers.Conv2D(256, 3, 1, "same")
    self.conv_c6_3x3 = layers.Conv2D(256, 3, 2, "same")
    self.conv_c7_3x3 = layers.Conv2D(256, 3, 2, "same")
    self.upsample_2x = layers.UpSampling2D(2)

  
  def call(self, images, training=False):
    c3_output, c4_output, c5_output = self.backbone(images, training=training)
    p3_output = self.conv_c3_1x1(c3_output)
    p4_output = self.conv_c4_1x1(c4_output)
    p5_output = self.conv_c5_1x1(c5_output)
    p4_output = p4_output + self.upsample_2x(p5_output)
    p3_output = p3_output + self.upsample_2x(p4_output)
    p3_output = self.conv_c3_3x3(p3_output)
    p4_output = self.conv_c4_3x3(p4_output)
    p5_output = self.conv_c5_3x3(p5_output)
    p6_output = self.conv_c6_3x3(c5_output)
    p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
    
    return p3_output, p4_output, p5_output, p6_output, p7_output

def build_head(output_filters, bias_init):
  head = keras.Sequential([keras.Input(shape=[None, None, 256])])
  kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
  for _ in range(4):
      head.add(keras.layers.Conv2D(256, 3, padding="same", kernel_initializer=kernel_init))
      head.add(keras.layers.ReLU())
  head.add(keras.layers.Conv2D(output_filters, 
                               3, 
                               1,
                               padding="same",
                               kernel_initializer=kernel_init,
                               bias_initializer=bias_init))
  
  return head

class RetinaNet(Model):
    def __init__(self, num_classes, backbone=None, **kwargs):
        super(RetinaNet, self).__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")


    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(tf.reshape(self.cls_head(feature), [N, -1, self.num_classes]))
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
    
        return tf.concat([box_outputs, cls_outputs], axis=-1)

def prepare_image(image):
  image, _, ratio = resize_and_pad_image(image, jitter=None)
  image = tf.keras.applications.resnet.preprocess_input(image)
  return tf.expand_dims(image, axis=0), ratio

def create_model(num_classes):
    resnet50_backbone = get_backbone()
    model = RetinaNet(num_classes, resnet50_backbone)

    return model