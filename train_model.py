"""Train one model — exact config that worked for models 1-3."""
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from data_loader import create_dataloaders

GEO_MODELS = {"15_GeoDualBranch", "16_GeoFiLM", "17_MultiTask-Geo"}


def compute_accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


def train_epoch(model, loader, optimizer, criterion, device, scaler, model_name, epoch_num):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    total_batches = len(loader)
    for step, batch in enumerate(loader):
        images, labels, coords = batch
        images, labels = images.to(device), labels.to(device)
        coords = coords.to(device)

        optimizer.zero_grad()
        with autocast():
            if model_name in GEO_MODELS:
                if model_name == "17_MultiTask-Geo":
                    country_logits, coord_pred = model(images, coords)
                    loss_country = criterion(country_logits, labels)
                    loss_coord = nn.MSELoss()(coord_pred, coords)
                    loss = loss_country + 0.1 * loss_coord
                    logits = country_logits
                else:
                    logits = model(images, coords)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        total_acc += compute_accuracy(logits.detach(), labels) * images.size(0)
        n += images.size(0)

        if (step + 1) % 20 == 0:
            print(f"    Epoch {epoch_num:2d} | Step {step+1:3d}/{total_batches} | loss: {total_loss/n:.4f} acc: {total_acc/n:.4f}", flush=True)

    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, model_name, neutral_coords=None):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for batch in loader:
        images, labels, coords = batch
        images, labels = images.to(device), labels.to(device)
        # For geo models during eval: replace real coords with neutral mean
        # so the model can't cheat using lat/lng. Training still gets real coords.
        if neutral_coords is not None and model_name in GEO_MODELS:
            coords = neutral_coords.expand(images.size(0), -1).to(device)
        else:
            coords = coords.to(device)

        with autocast():
            if model_name in GEO_MODELS:
                if model_name == "17_MultiTask-Geo":
                    country_logits, _ = model(images, coords)
                    loss = criterion(country_logits, labels)
                    logits = country_logits
                else:
                    logits = model(images, coords)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total_acc += compute_accuracy(logits, labels) * images.size(0)
        n += images.size(0)
    return total_loss / n, total_acc / n


def train_and_eval(model_name, build_fn, data_root, device, epochs=50, batch_size=32, lr=0.001, image_size=256):
    print(f"\n{'='*60}", flush=True)
    print(f"Training: {model_name}", flush=True)
    print(f"{'='*60}", flush=True)

    train_loader, valid_loader, test_loader, countries, _ = create_dataloaders(
        data_root, image_size=image_size, batch_size=batch_size, num_workers=4
    )

    # Compute training set mean lat/lng for neutral geo eval
    neutral_coords = None
    if model_name in GEO_MODELS:
        all_lats, all_lngs = [], []
        for batch in train_loader:
            _, _, coords = batch
            all_lats.append(coords[:, 0].mean().item())
            all_lngs.append(coords[:, 1].mean().item())
        mean_lat = sum(all_lats) / len(all_lats)
        mean_lng = sum(all_lngs) / len(all_lngs)
        neutral_coords = torch.tensor([mean_lat, mean_lng], dtype=torch.float32).to(device)
        print(f"Neutral eval coords: ({mean_lat:.2f}, {mean_lng:.2f})", flush=True)

    model = build_fn().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    best_valid_acc = 0.0
    best_state = None

    t0 = time.time()
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler, model_name, epoch + 1)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device, model_name, neutral_coords)
        scheduler.step()

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | valid_loss: {valid_loss:.4f} valid_acc: {valid_acc:.4f}", flush=True)

    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f}s ({train_time/60:.1f}m)", flush=True)

    model.load_state_dict(best_state)
    _, train_acc = evaluate(model, train_loader, criterion, device, model_name, neutral_coords)
    _, valid_acc = evaluate(model, valid_loader, criterion, device, model_name, neutral_coords)
    _, test_acc = evaluate(model, test_loader, criterion, device, model_name, neutral_coords)

    print(f"Final — Train: {train_acc:.4f} | Valid: {valid_acc:.4f} | Test: {test_acc:.4f}", flush=True)
    print(f"Best Valid: {best_valid_acc:.4f}", flush=True)

    return {
        "model": model_name,
        "params": n_params,
        "train_acc": round(train_acc, 4),
        "valid_acc": round(valid_acc, 4),
        "test_acc": round(test_acc, 4),
        "best_valid_acc": round(best_valid_acc, 4),
        "time_sec": round(train_time, 1),
        "epochs": epochs,
        "batch_size": batch_size,
    }
