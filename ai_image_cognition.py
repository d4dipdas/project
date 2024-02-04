import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# Load a pre-trained PyTorch model
model = models.resnet50(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
])

# Function to classify image
def classify_image(image):
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class and probability
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class_index = torch.argmax(probabilities).item()

    return predicted_class_index, probabilities[predicted_class_index].item()

# Function to get the label for a predicted class index
def get_label(class_index):
    return labels[class_index]

# Open a connection to the camera (assuming camera index 0)
cap = cv2.VideoCapture(0)

# Download ImageNet labels
LABELS_URL = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
response = requests.get(LABELS_URL)
labels = response.json()

# Load Haar Cascade Classifier for object detection
haarcascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Main loop to capture frames from the camera
while True:
    # Capture frame-by-frame
    ret, frame = cap.read(0)

    # Convert the OpenCV BGR image to RGB (PIL Image format)
    if frame is not None:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Classify the image
        class_index, confidence = classify_image(pil_image)
        predicted_label = get_label(class_index)

        # Object detection using Haar Cascade Classifier
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rectangles = haarcascade_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected objects
        for (x, y, w, h) in rectangles:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the result on the frame
        cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
