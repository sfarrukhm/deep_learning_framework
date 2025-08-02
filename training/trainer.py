import torch
import copy
from tqdm import tqdm

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs, save_path='best_model.pth'):
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        print(f"\n📘 Epoch {epoch+1}/{epochs}")

        # ---- Training Phase ----
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc="🟢 Training", leave=False)
        for xb, yb in train_bar:
            xb, yb = xb.to(device), yb.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)

        # ---- Validation Phase ----
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc="🔵 Validating", leave=False)
        with torch.no_grad():
            for xb, yb in val_bar:
                xb, yb = xb.to(device), yb.to(device).float().unsqueeze(1)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())
        val_loss /= len(val_loader)

        # ---- Save best model ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)
            print(f"✅ Best Model Saved | Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        else:
            print(f"📉 Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Load best weights back into model
    model.load_state_dict(best_model_wts)
