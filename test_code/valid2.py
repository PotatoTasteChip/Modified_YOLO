from ultralytics import YOLO

# Load a model
#model = YOLO("yolo11.pt")  # load an official model
# model = YOLO("yolo11x.yaml") 
# model = YOLO("/home/vprism1/beaver_ws/dataset/model/best_v11x.pt")  # load a custom model

model = YOLO("/home/vprism_3/modified_YOLO5/ultralytics/cfg/models/11/yolo11-fpsa-l.yaml") 
model = YOLO("/home/vprism_3/modified_YOLO5/test_code/runs/train/modified_YOLO5_large_202502122/weights/best.pt")  # load a custom model


# Validate the model
metrics = model.val(
    data="/home/vprism_3/modified_YOLO5_val/test_code/data_yaml/B_21_test.yaml",
    device='cpu',
    name="fpsa_val_l_315",
    )  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category