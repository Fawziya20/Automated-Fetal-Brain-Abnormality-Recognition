import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np

# Load YOLO model
model = YOLO('runs\\detect\\train\\weights\\best.pt')

# Function to make predictions and annotate the image
def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    results = predict(chosen_model, img, classes, conf=conf)

    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img, results

# Streamlit App
def main():
    st.title("Fetal Brain Abnormality Prediction")

    # Upload image through Streamlit
    uploaded_image = st.file_uploader("Upload an ultrasound image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image")

        # Convert the uploaded image to a format that your model can use
        image = Image.open(uploaded_image).convert('RGB')
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Process the image and make predictions
        result_img,_ = predict_and_detect(model, image_np, classes=[0, 1], conf=0.5)

        # Display the annotated image
        st.image(result_img)
        image_name = uploaded_image.name.split('.')[0]  # Extracting the name without extension
        st.write(image_name)

# Run the Streamlit app
if __name__ == "__main__":
    main()
