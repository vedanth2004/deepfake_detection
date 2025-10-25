import torch
import cv2
from torchvision import models, transforms
import torch.nn as nn
import numpy as np

# -------------------
# 1️⃣ Setup
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_SIZE = (224, 224)

# -------------------
# 2️⃣ Define same model architecture
# -------------------
def get_model(name="resnet18", num_classes=2, pretrained=False):
    if name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# -------------------
# 3️⃣ Load trained model
# -------------------
model = get_model("resnet18").to(device)
model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
model.eval()
print("✅ Model loaded successfully!")

# -------------------
# 4️⃣ Define transforms (same as training)
# -------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------------------
# 5️⃣ Video inference function (NO OUTPUT VIDEO)
# -------------------
def detect_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Cannot open video file.")
        return

    frame_count = 0
    real_votes = 0
    fake_votes = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_t = transform(rgb).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            outp = model(img_t)
            prob = torch.softmax(outp, dim=1)[0].cpu().numpy()
            pred = prob.argmax()

        label = "REAL" if pred == 0 else "FAKE"
        conf = prob[pred]

        # Count predictions for overall final result
        if label == "REAL":
            real_votes += 1
        else:
            fake_votes += 1

        frame_count += 1

        # Optional live preview
        cv2.putText(frame, f"{label} ({conf:.2f})", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0) if label=="REAL" else (0,0,255), 3)
        cv2.imshow("Deepfake Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # -------------------
    # Final decision based on all frames
    # -------------------
    final_label = "REAL" if real_votes > fake_votes else "FAKE"
    print(f"\n✅ Video analyzed successfully!")
    print(f"📊 Total Frames: {frame_count}")
    print(f"🟢 Real Frames: {real_votes}, 🔴 Fake Frames: {fake_votes}")
    print(f"🏁 Final Prediction: {final_label}")

# -------------------
# 6️⃣ Run on a video file
# -------------------
input_video = "hi_.mp4"  # change to your video path
detect_from_video(input_video)
