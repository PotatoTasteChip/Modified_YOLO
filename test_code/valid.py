from ultralytics import YOLO
import numpy as np
import pandas as pd

# Load the model
model = YOLO("/home/vprism1/beaver_ws/dataset/model/best_v8x.pt")  # Load your custom model

# Path to validation dataset
data_path = "/home/vprism1/beaver_ws/validation_ws/ultralytics/test_code/data_yaml/B_21_test.yaml"

# Confidence values
#confidence_values = np.concatenate([[0.001], np.arange(0.01, 1.01, 0.01)])
confidence_values = [0.542]
# Path to save the results
output_file = "/home/vprism1/compare_yolo/temp/confidence_metrics_v11_0_542_2_iou.csv"

# Initialize or load results DataFrame
try:
    results_df = pd.read_csv(output_file)  # Try to load existing results
    print(f"Loaded existing results from '{output_file}'.")
except FileNotFoundError:
    results_df = pd.DataFrame(columns=['Confidence', 'Precision', 'Recall', 'mAP50', 'mAP50-95'])
    print(f"No existing results file found. Starting fresh.")

# Loop over confidence values
for conf in confidence_values:
    print(f"Validating with confidence: {conf}")
    
    # Check if this confidence value has already been processed
    if conf in results_df['Confidence'].values:
        print(f"Confidence {conf} already processed. Skipping.")
        continue
    
    # Validate the model with the current confidence value
    metrics = model.val(
        data=data_path,
        device=1,
        conf=conf,
        iou=0.6,
        name=f"/home/vprism1/compare_yolo/vx_temp/yolov8x_{conf}"
    )
    
    # Extract the summary metrics
    row = {
        'Confidence': conf,
        'Precision': metrics.box.mp,  # Mean Precision
        'Recall': metrics.box.mr,    # Mean Recall
        'mAP50': metrics.box.map50, # Mean AP@0.5
        'mAP50-95': metrics.box.map  # Mean AP@0.5:0.95
    }
    
    # Append the result to the DataFrame
    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    
    # Save the updated DataFrame to the CSV file
    results_df.to_csv(output_file, index=False)
    print(f"Saved results for confidence {conf} to '{output_file}'.")

print("All results have been processed and saved.")
