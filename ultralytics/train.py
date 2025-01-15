from ultralytics import YOLO
from torchinfo import summary

model = YOLO('/home/vprism_3/modified_YOLO5/ultralytics/cfg/models/11/yolo11-fpsa.yaml')

summary(model.model, input_size=(1, 3, 640, 640))
