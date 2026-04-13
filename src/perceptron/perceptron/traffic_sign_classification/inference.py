import torch
import os
import cv2
from PIL import Image
from torchvision import transforms

from .network import ResNet18


CLASS_NAMES = {
    0: "Stop",
    1: "Turn right",
    2: "Turn left",
    3: "Ahead only",
    4: "Roundabout mandatory",
}


def load_model(model_path, device):
    """Load a ResNet-18 checkpoint and return the model in evaluation mode.

    Handles three checkpoint formats:
      - dict with a "model" key (state-dict wrapped in a training snapshot).
      - plain state-dict (keys map directly to model parameters).
      - a serialised full model object (saved with torch.save(model, ...)).

    Args:
        model_path: Path to the .pth checkpoint file.
        device:     torch.device to map the weights onto.

    Returns:
        ResNet18 model ready for inference.
    """
    model = ResNet18(num_classes=len(CLASS_NAMES)).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        # Full model object serialised directly
        model = checkpoint.to(device)

    model.eval()
    return model


def inference(model, device, img):
    """Classify a single PIL Image and return the predicted sign class name.

    Pre-processes the image to 32x32 with ImageNet-style normalisation to match
    the training transform, runs a forward pass under torch.no_grad(), and maps
    the argmax logit to a human-readable class name via CLASS_NAMES.

    Args:
        model:  Loaded ResNet-18 model (from load_model).
        device: torch.device the model lives on.
        img:    PIL Image (RGB) of the sign crop.

    Returns:
        String class name, e.g. "Stop" or "Turn left".
    """
    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_idx = int(torch.argmax(probabilities, dim=1).item())

    return CLASS_NAMES[predicted_idx]


if __name__ == "__main__":
    model_path = '/home/hdqquang/Projects/MTRX5700_Assignment_2/Vision Task/checkpoint/ckpt_rmsprop_lr_0.005_bs_64_ep_100_True.pth'
    image_path = 'dinh test square images/output_signs'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    for img in os.listdir(image_path):
        img_path = os.path.join(image_path, img)
        img = Image.open(img_path).convert("RGB")
        predicted_class = inference(model, device, img)
        print(f"Image: {img} - Predicted Class: {predicted_class}")

        img = cv2.imread(img_path)
        img = cv2.resize(img, (400, 400))
        cv2.imshow(f"Prediction: {predicted_class}", img)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()