# File that loads a trained model and predicts the class of an image

import torch
from cv2 import imread
from config import get_config, init_config
from data_preparation import DataPreparation
from model import Vision_Transformer

if __name__ == "__main__":
    init_config()
    config = get_config()
    vit_model = config["MODEL"]["VIT"]["PRE_TRAINED"]
    # load the trained model
    model = Vision_Transformer(pretrained_model=vit_model, reference_image=False).to("cpu")
    model.load_state_dict(torch.load("models/all_angles/model.pth", map_location="cpu"))

    # Set the model to evaluation mode
    model.eval()

    # Load transformer
    data_preparation = DataPreparation()
    _, _, _, transform, _ = data_preparation.create_dataloaders()

    # Load the image
    image_path = "images/WT-07/2024-02-13_15_08_WT_07_SVIV02.jpg"  # Replace with your image path

    image = imread(image_path)
    image = transform(image=image)["image"].unsqueeze(0)  # Apply the transform and add batch dimension
    image = image.to("cpu")  # Move to CPU if necessary

    # Predict the class
    result = model(image).squeeze()  # Remove batch dimension

    mean = 0.036641769111156464
    std = 7.633823871612549

    # Normalize the output
    result = (result - mean) / std

    print(f"Predicted class: {result.item()}")  # Print the predicted class


