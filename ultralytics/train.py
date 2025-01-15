from ultralytics import YOLO
from torchinfo import summary

model = YOLO('/home/vprism1/modified_YOLO4/ultralytics/cfg/models/11/yolo11-vprism.yaml')

summary(model.model, input_size=(1, 3, 640, 640))
