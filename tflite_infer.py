import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

######
### Load TFLite model
######
interpreter = tf.lite.Interpreter(model_path="resnet34_peoplenet_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

### for Debug
# print(input_details)
# [{'name': 'serving_default_input_1:0', 'index': 0, 'shape': array([  1, 544, 960,   3], dtype=int32), 'shape_signature': array([  1, 544, 960,   3], dtype=int32), 'dtype': <class 'numpy.int8'>, 'quantization': (0.003921565134078264, -128), 'quantization_parameters': {'scales': array([0.00392157], dtype=float32), 'zero_points': array([-128], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
# print(output_details)
# [{'name': 'PartitionedCall:1', 'index': 178, 'shape': array([ 1, 34, 60, 12], dtype=int32), 'shape_signature': array([ 1, 34, 60, 12], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}, {'name': 'PartitionedCall:0', 'index': 176, 'shape': array([ 1, 34, 60,  3], dtype=int32), 'shape_signature': array([ 1, 34, 60,  3], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]

######
### Generate Input Tensor
######
### Load Image
#img = cv2.imread("input.jpg")
img = cv2.imread("sample_1080p_h265_frame_input.png")
### image information
height, width, channel = img.shape

### Resize and convert to int8
img_resized = cv2.resize(img, (960, 544))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
# uint8
# print(img_rgb.dtype) 
# 255 / 0
# print(img_rgb.max(), img_rgb.min()) 
img_signed_int8 = img_rgb - 128
### make Input Tensor
predict_img = np.expand_dims(img_signed_int8, axis=0).astype("int8")
# int8
# print(predict_img.dtype)
# 127 / -128
# print(predict_img.max(), predict_img.min())
# (1, 544, 960, 3)
# print(predict_img.shape) 

######
### set Tensor <<-- predict_img
######
# set input
interpreter.set_tensor(input_details[0]['index'], predict_img)

######
### infer (invoke)
######
interpreter.invoke()

# get output
output_data_bbox = interpreter.get_tensor(output_details[0]['index'])
output_data_class = interpreter.get_tensor(output_details[1]['index'])

### check Output[0] (for bbox?)
# Debug
# (1, 34, 60, 12)
# print(output_data_bbox.shape) 
# print(output_data_bbox)

### check Output[1] (for class?)
# Debug
# (1, 34, 60, 3)
# print(output_data_class.shape) 
# print(output_data_class)

# Debug
# print(output_data_class.max(), output_data_class.min()) # 0.5 0.0

######
### reference
######
# -->> https://github.com/patterson163/TensorRT-PeopleNet/blob/master/sources/main.py
# -->> https://github.com/Kwull/deepstream-4.0.1/blob/master/sources/libs/nvdsinfer_customparser/nvdsinfer_custombboxparser.cpp

class PeopleNetPostProcess(object):
    def __init__(self, width, height, score_threshold=0.5):
        self.image_width = width
        self.image_height = height

        self.model_h = 544
        self.model_w = 960

        self.strideX = 16.0
        self.strideY = 16.0

        self.bboxNormX = 35.0
        self.bboxNormY = 35.0

        self.grid_h = int(self.model_h / self.strideY)
        self.grid_w = int(self.model_w / self.strideX)
        self.grid_size = self.grid_h * self.grid_w
        # debug
        # print(self.grid_h, self.grid_w, self.grid_size)

        ### make Grid Information
        self.grid_centers_w = []
        self.grid_centers_h = []

        for i in range(self.grid_h):
            value = (i * self.strideY + 0.5) / self.bboxNormY
            self.grid_centers_h.append(value)

        for i in range(self.grid_w):
            value = (i * self.strideX + 0.5) / self.bboxNormX
            self.grid_centers_w.append(value)

        # debug
        # print(self.grid_centers_h)
        # print(self.grid_centers_w)

        ### thresholds
        self.min_confidence = score_threshold
        self.num_of_totalclasses = 3

    def applyBoxNorm(self, o1, o2, o3, o4, w, h):
        o1 = (self.grid_centers_w[w] - o1) * self.bboxNormX
        o2 = (self.grid_centers_h[h] - o2) * self.bboxNormY
        o3 = (o3 + self.grid_centers_w[w]) * self.bboxNormX
        o4 = (o4 + self.grid_centers_h[h]) * self.bboxNormY
        return o1, o2, o3, o4

    def change_model_size_to_real(self, model_size, type):
        real_size = 0
        if type == 'x':
            real_size = (model_size / float(self.model_w)) * self.image_width
        elif type == 'y':
            real_size = (model_size / float(self.model_h)) * self.image_height
        real_size = int(real_size)
        return real_size

    def start(self, feature_bbox, feature_scores, classes=[0]):
        boundingboxes = []
        self.analysis_classeds = classes

        ### for each-class
        for c in self.analysis_classeds: #range(self.num_of_totalclasses):
            ### search in Grid HxW
            for h in range(self.grid_h):
                for w in range(self.grid_w):
                    ### check probability
                    class_probability = feature_scores[h][w][c]
                    if class_probability >= self.min_confidence:  
                        print("found", h, w, c)              
                        # get Bounding Box Info (for-all classes)
                        bbox_raw = feature_bbox[h][w]
                        # for Single-Class
                        bbox_idx_start = c * 4
                        bbox_idx_end = (c+1) * 4
                        bbox_part = bbox_raw[bbox_idx_start:bbox_idx_end]
                        # print Bounding Box Info
                        # debug
                        # print(h, w, c, bbox_part)
                        ### get Offset BBOX from Center of Grid
                        o1, o2, o3, o4 = bbox_part
                        # grid-center - o1 = LEFT
                        # grid-center + o3 = RIGHT
                        # grid-center - o2 = TOP
                        # grid-center + o4 = BOTTOM
                        ### get POS BBOX for resized image
                        o1, o2, o3, o4 = self.applyBoxNorm(o1, o2, o3, o4, w, h)
                        xmin_model = int(o1)
                        ymin_model = int(o2)
                        xmax_model = int(o3)
                        ymax_model = int(o4)
                        # print Normalized Bounding Box
                        # debug
                        # print(h, w, c, xmin_model, ymin_model, xmax_model, ymax_model)  
                        ### get POS BBOX for non-resized (original) image
                        xmin_image = self.change_model_size_to_real(xmin_model, 'x')
                        ymin_image = self.change_model_size_to_real(ymin_model, 'y')
                        xmax_image = self.change_model_size_to_real(xmax_model, 'x')
                        ymax_image = self.change_model_size_to_real(ymax_model, 'y')
                        # print Normalized Bounding Box
                        # debug
                        # print(h, w, c, xmin_image, ymin_image, xmax_image, ymax_image)  
                        # Put BoundingBox 
                        boundingbox = (xmin_image, ymin_image, xmax_image, ymax_image)
                        boundingboxes.append(boundingbox)

        return boundingboxes

### call decoder
post_process = PeopleNetPostProcess(width, height)
boundingboxes = post_process.start(output_data_bbox[0], output_data_class[0], classes=[0, 1, 2])

for bbox in boundingboxes:
    left, top, right, bottom = bbox
    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

cv2.imwrite('sample_1080p_h265_frame_output.png', img)


