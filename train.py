import torch
import torch.nn as nn
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split

print("Parsing saved arrays...")
DATA_DIR = "gesture_data"
MODEL_DIR = "models"
GESTURES = ["open_hand", "no_gesture", "closed_hand"]

X, y = [], []
for label, gesture in enumerate(GESTURES):
    path = os.path.join(DATA_DIR, f"{gesture}.npy")
    
    # Check fallback names (if applicable)
    if not os.path.exists(path):
        alt = {"open_hand": "open_palm"}.get(gesture, gesture)
        path = os.path.join(DATA_DIR, f"{alt}.npy")
    
    if not os.path.exists(path):
        print(f"Warning: Missing data file for {gesture} expected at {path}")
        continue
        
    data = np.load(path)
    X.append(data)
    y += [label] * len(data)
    print(f"  Loaded {gesture}: Shape context {data.shape}")

X = np.vstack(X)
y = np.array(y)
print(f"Complete Matrix Form: Loaded {len(X)} matrices across {len(GESTURES)} target categories.")

# Train/Test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

class MLP(nn.Module):
    """Multi-Layer Perceptron neural structural layer mapping coordinates to predictions."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x): 
        return self.net(x)

INPUT_DIM = X_train.shape[1]
model = MLP(INPUT_DIM, len(GESTURES))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
criterion = nn.CrossEntropyLoss()

log = {"epochs": [], "gestures": GESTURES}
best_acc = 0

print("\nStarting optimization loop (150 epochs)...")
for epoch in range(150):
    model.train()
    optimizer.zero_grad()
    
    loss = criterion(model(X_train), y_train)
    loss.backward()
    
    optimizer.step()
    scheduler.step()

    # Track classification metrics every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            preds = model(X_test).argmax(1)
            overall = (preds == y_test).float().mean().item()
            
            per_class = []
            for c in range(len(GESTURES)):
                mask = y_test == c
                acc = (preds[mask] == y_test[mask]).float().mean().item() if mask.sum() > 0 else 0.0
                per_class.append(round(acc * 100, 1))
                
        log["epochs"].append({
            "epoch": epoch + 1,
            "overall": round(overall * 100, 1),
            "per_class": per_class
        })
        
        print(f"Epoch {epoch+1:03d}/150 | Val Acc: {overall*100:.1f}% | Per-Class: {per_class}")
        
        # Save checkpoints
        if overall > best_acc:
            best_acc = overall
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))

print(f"\nTraining done! Best Validation Accuracy achieved: {best_acc*100:.2f}%")
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth"), weights_only=True))

# Export log metadata for UI chart mapping component
with open(os.path.join(MODEL_DIR, "training_log.json"), "w") as f:
    json.dump(log, f)

print("Exporting production model to ONNX runtime file format...")
model.eval()
dummy = torch.zeros(1, INPUT_DIM)
out_path = "gesture_model.onnx"

if os.path.exists(out_path): 
    os.remove(out_path)
    
with torch.no_grad():
    torch.onnx.export(
        model, dummy, out_path,
        export_params=True, 
        do_constant_folding=True,
        input_names=["features"], 
        output_names=["logits"], 
        dynamo=False
    )
    
print("ONNX binary successfully written to project root. Run server to test!")
