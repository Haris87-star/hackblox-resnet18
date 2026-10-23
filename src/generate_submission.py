import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from pathlib import Path
from src.train import IntelSceneResNet18

PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# Load trained model
model = IntelSceneResNet18(num_classes=len(CLASSES))
weights_path = PROJECT_ROOT / "best_resnet18_scene.pth"
model.load_state_dict(torch.load(str(weights_path), map_location=DEVICE))
model.to(DEVICE)
model.eval()

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_kaggle_predictions():
    test_dirs = [
        PROJECT_ROOT / "scene_data" / "seg_pred" / "seg_pred",
        PROJECT_ROOT / "scene_data" / "unlabeled",
        PROJECT_ROOT / "scene_data" / "test"
    ]
    
    test_dir = next((d for d in test_dirs if d.exists()), None)
    if not test_dir:
        print("⚠️ No test folder found. Place competition test images in 'scene_data/unlabeled' or 'scene_data/seg_pred'.")
        return

    image_files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    print(f"🔍 Found {len(image_files)} test images in {test_dir}. Generating predictions...")

    records = []
    with torch.no_grad():
        for fname in image_files:
            img_path = test_dir / fname
            img = Image.open(img_path).convert("RGB")
            img_t = torch.as_tensor(eval_tf(img)).unsqueeze(0).to(DEVICE)

            # Test-Time Augmentation (Original + Horizontal Flip)
            l1, _ = model(img_t)
            l2, _ = model(torch.flip(img_t, dims=[3]))
            probs = torch.softmax(l1, dim=1) + torch.softmax(l2, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())

            records.append({
                "image_name": fname,
                "label": pred_idx,
                "predicted_scene": CLASSES[pred_idx]
            })

    df = pd.DataFrame(records)
    submission_path = PROJECT_ROOT / "submission.csv"
    df.to_csv(submission_path, index=False)
    print(f"✅ Generated '{submission_path}' ready to upload to Kaggle Leaderboard!")

if __name__ == "__main__":
    generate_kaggle_predictions()