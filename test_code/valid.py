from ultralytics import YOLO

# Load a model
#model = YOLO("yolo11.pt")  # load an official model
# model = YOLO("yolo11x.yaml") 
# model = YOLO("/home/vprism1/beaver_ws/dataset/model/best_v11x.pt")  # load a custom model

model = YOLO("yolov8x.yaml") 
model = YOLO("/home/vprism1/beaver_ws/dataset/model/best_v8x.pt")  # load a custom model

# model = YOLO("/home/vprism1/beaver_ws/dataset/model/fpsa/yolo11-fpsa.yaml") 
# model = YOLO("/home/vprism1/beaver_ws/dataset/model/fpsa/weights/best.pt")  # load a custom model


# Validate the model
metrics = model.val(
    data="/home/vprism1/modified_YOLO4/test_code/data_yaml/B_21_test.yaml",
    device=1,
    name="yolov8x",
    conf = 0.6,
    )  # no arguments needed, dataset and settings remembered

metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category