import joblib
import pandas as pd

# Load the trained model
model = joblib.load('threshold_model.pkl')

def predict_threshold(distance, angle, object_width, object_height, object_depth, layer_height):
    # Create input DataFrame
    input_data = pd.DataFrame([[distance, angle, object_width, object_height, object_depth, layer_height]],
                              columns=['distance', 'angle', 'object_width', 'object_height', 'object_depth', 'layer_height'])

    # Predict threshold
    predicted_threshold = model.predict(input_data)
    return predicted_threshold[0]

if __name__ == "__main__":
    # Get user input
    distance = float(input("Enter the distance from the camera to the printer (in cm): "))
    angle = float(input("Enter the angle of the camera relative to the printer (in degrees): "))
    object_width = float(input("Enter the width of the object being printed (in cm): "))
    object_height = float(input("Enter the height of the object being printed (in cm): "))
    object_depth = float(input("Enter the depth of the object being printed (in cm): "))
    layer_height = float(input("Enter the height of each layer (in mm): "))

    # Predict threshold
    threshold = predict_threshold(distance, angle, object_width, object_height, object_depth, layer_height)
    print(f"Predicted threshold value: {threshold}")
