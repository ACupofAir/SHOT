import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import network
import matplotlib.pyplot as plt
import numpy as np

VISDA_CLASS_MAP = {
    0: "aeroplane",
    1: "bicycle",
    2: "bus",
    3: "car",
    4: "horse",
    5: "knife",
    6: "motorcycle",
    7: "person",
    8: "plant",
    9: "skateboard",
    10: "train",
    11: "truck",
}

def load_model(model_dir, net_type, bottleneck_dim, class_num, classifier_type, layer_type):
    # Load feature extractor
    if net_type[0:3] == 'res':
        netF = network.ResBase(res_name=net_type).cuda()
    elif net_type[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=net_type).cuda()

    # Load bottleneck and classifier
    netB = network.feat_bottleneck(type=classifier_type, feature_dim=netF.in_features, bottleneck_dim=bottleneck_dim).cuda()
    netC = network.feat_classifier(type=layer_type, class_num=class_num, bottleneck_dim=bottleneck_dim).cuda()

    # Load weights
    netF.load_state_dict(torch.load(f"{model_dir}/source_F.pt", weights_only=True))
    netB.load_state_dict(torch.load(f"{model_dir}/source_B.pt", weights_only=True))
    netC.load_state_dict(torch.load(f"{model_dir}/source_C.pt", weights_only=True))

    netF.eval()
    netB.eval()
    netC.eval()

    return netF, netB, netC

def preprocess_image(image_path, resize_size=256, crop_size=224):
    transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.cuda()

def infer(image_path, model_dir, net_type='resnet101', bottleneck_dim=256, class_num=65, classifier_type='bn', layer_type='wn'):
    """
    Returns:
        predicted class and the output probabilities
    """
    # Load models
    netF, netB, netC = load_model(model_dir, net_type, bottleneck_dim, class_num, classifier_type, layer_type)
    image = preprocess_image(image_path)

    # Perform inference
    with torch.no_grad():
        features = netB(netF(image))
        outputs = netC(features)
        _, predicted = torch.max(outputs, 1)

    return predicted.item(), outputs.cpu().numpy()[0]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference Script for SHOT")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory containing the trained model files")
    parser.add_argument('--net', type=str, default='resnet101', help="Network type (e.g., resnet50, vgg16)")
    parser.add_argument('--bottleneck', type=int, default=256, help="Bottleneck dimension")
    parser.add_argument('--class_num', type=int, default=12, help="Number of classes")
    parser.add_argument('--classifier', type=str, default="bn", help="Classifier type (e.g., bn, ori)")
    parser.add_argument('--layer', type=str, default="wn", help="Layer type (e.g., linear, wn)")

    args = parser.parse_args()

    # Perform inference
    result, prob = infer(
        image_path=args.image_path,
        model_dir=args.model_dir,
        net_type=args.net,
        bottleneck_dim=args.bottleneck,
        class_num=args.class_num,
        classifier_type=args.classifier,
        layer_type=args.layer
    )
    
    # prob is 1x12 numpy array, draw its softmax pdf, and mark the value on the image
    # Compute softmax probabilities
    softmax_probs = np.exp(prob) / np.sum(np.exp(prob))

    # Plot the softmax probabilities
    plt.figure(figsize=(10, 5))
    plt.bar(VISDA_CLASS_MAP.values(), softmax_probs, color='skyblue')
    plt.xlabel("Classes")
    plt.ylabel("Probability")
    plt.title("Softmax Probability Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Mark the predicted class on the image
    predicted_class = VISDA_CLASS_MAP[result]
    image = Image.open(args.image_path).convert("RGB")
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Predicted: {predicted_class} ({softmax_probs[result]:.2f})", fontsize=16)
    plt.show()
    

