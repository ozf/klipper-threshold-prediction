import joblib
import pandas as pd
import sys

# Load the trained model
model = joblib.load('threshold_model.pkl')

def predict_threshold(distance, angle_vertical, angle_horizontal, object_width, object_height, object_depth, layer_height):
    # Create input DataFrame
    input_data = pd.DataFrame([[distance, angle_vertical, angle_horizontal, object_width, object_height, object_depth, layer_height]],
                              columns=['distance', 'angle_vertical', 'angle_horizontal', 'object_width', 'object_height', 'object_depth', 'layer_height'])

    # Predict threshold
    predicted_threshold = model.predict(input_data)
    return predicted_threshold[0]

if __name__ == "__main__":
    # Read parameters from command-line arguments
    distance = float(sys.argv[1])
    angle_vertical = float(sys.argv[2])
    angle_horizontal = float(sys.argv[3])
    object_width = float(sys.argv[4])
    object_height = float(sys.argv[5])
    object_depth = float(sys.argv[6])
    layer_height = float(sys.argv[7])

    # Predict threshold
    threshold = predict_threshold(distance, angle_vertical, angle_horizontal, object_width, object_height, object_depth, layer_height)
    print(threshold)
