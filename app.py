import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Model

class LinearProbingModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# Load model

device = torch.device("cpu")

model = LinearProbingModel()

model.load_state_dict(
    torch.load(
        "final_pneumonia_classifier.pth",
        map_location=device
    )
)

model.eval()

# Transform

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485],
        std=[0.229]
    )
])

classes = {
    0:"NORMAL",
    1:"PNEUMONIA"
}

# UI

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺"
)

st.title("🩺 Pneumonia Detection System")

st.write(
    "Upload a Chest X-Ray image."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded X-Ray",
        use_container_width=True
    )

    img = transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(img)

        probs = torch.softmax(
            output,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            1
        )

    prediction = classes[pred.item()]

    st.success(
        f"Prediction: {prediction}"
    )

    st.info(
        f"Confidence: {confidence.item()*100:.2f}%"
    )