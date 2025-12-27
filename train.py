import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json


DATASET_DIR = "emotion_dataset"
BATCH_SIZE = 16
EPOCHS = 35
LR = 1e-3
SEQUENCE_LENGTH = 30



def load_npz_dataset():
    features = []
    labels = []

    for fname in os.listdir(DATASET_DIR):
        if fname.endswith(".npz"):
            path = os.path.join(DATASET_DIR, fname)
            data = np.load(path)

            seq = data["features"]   
            label = int(data["label"])

            features.append(seq)
            labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    print(f"Loaded {len(features)} sequences.")
    return features, labels



class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        return context


class EmotionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.attn = Attention(hidden_dim * 2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attn(lstm_out)
        return self.fc(context)



def main():

   
    X, y = load_npz_dataset()

    feature_dim = X.shape[2]

   
    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, feature_dim)
    scaler.fit(X_reshaped)
    X_scaled = scaler.transform(X_reshaped).reshape(X.shape)

   
    os.makedirs("models", exist_ok=True)
    np.save("models/scaler_mean.npy", scaler.mean_)
    np.save("models/scaler_scale.npy", scaler.scale_)

   
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=True, stratify=y
    )

    train_dataset = EmotionDataset(X_train, y_train)
    val_dataset = EmotionDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

   
    class_map_path = os.path.join(DATASET_DIR, "class_map.json")
    if os.path.exists(class_map_path):
        with open(class_map_path, "r") as f:
            class_map = json.load(f)
        num_classes = len(class_map)
    else:
        num_classes = 5

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    model = EmotionNet(feature_dim, hidden_dim=256, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

     
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model(Xb)
                correct += (preds.argmax(1) == yb).sum().item()
                total += yb.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.3f} | Val Acc: {acc:.2f}%")

   #save model
    save_path = "models/emotion_model_npz.pt"
    torch.save(model.state_dict(), save_path)
    print("\n🔥 Model saved to:", save_path)
    print("🔥 Scaler saved in models/")
    print("Training complete!")


if __name__ == "__main__":
    main()
