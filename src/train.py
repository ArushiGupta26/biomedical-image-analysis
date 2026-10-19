import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from src.data import make_loaders
from src.model import SmallCNN

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses, y_true, y_pred = [], [], []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.view(-1).long().to(device)
        if training: optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(1).detach().cpu().numpy().tolist())
    return float(np.mean(losses)), np.array(y_true), np.array(y_pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    train_loader, val_loader, test_loader = make_loaders(args.batch_size)
    model = SmallCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_f1 = -1
    Path("outputs").mkdir(exist_ok=True)
    for epoch in range(args.epochs):
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, y_val, p_val = run_epoch(model, val_loader, loss_fn, device)
        val_f1 = f1_score(y_val, p_val, average="macro", zero_division=0)
        print(f"epoch={epoch+1} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "outputs/best_model.pt")
    model.load_state_dict(torch.load("outputs/best_model.pt", map_location=device))
    test_loss, y_test, p_test = run_epoch(model, test_loader, loss_fn, device)
    metrics = {
        "seed": args.seed, "epochs": args.epochs, "device": str(device),
        "test_loss": test_loss,
        "test_accuracy": float(accuracy_score(y_test, p_test)),
        "test_macro_f1": float(f1_score(y_test, p_test, average="macro", zero_division=0)),
        "classification_report": classification_report(y_test, p_test, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, p_test).tolist(),
    }
    with open("outputs/metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    print(json.dumps({k:v for k,v in metrics.items() if k not in ("classification_report", "confusion_matrix")}, indent=2))
if __name__ == "__main__":
    main()
