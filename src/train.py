import torch
import torch.nn as nn
from .data import ArgoverseVehicleDataset, load_polars_dataframe
from .models import TransformerMotionPredictor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class SmoothTrajectoryLoss(nn.Module):
    def __init__(self, ade_weight=1.0, fde_weight=2.0, smoothing_weight=0.2):
        super().__init__()
        self.ade_weight = ade_weight
        self.fde_weight = fde_weight
        self.smoothing_weight = smoothing_weight
        self.huber_loss = nn.SmoothL1Loss()

    def forward(self, preds, targets):
        ade_loss = self.huber_loss(preds, targets)
        fde_loss = self.huber_loss(preds[:, -1, :], targets[:, -1, :])
        loss = (self.ade_weight * ade_loss) + (self.fde_weight * fde_loss)

        start_loss = torch.norm(preds[:, 0, :] - targets[:, 0, :], dim=-1).mean()
        loss += 2.0 * start_loss

        if preds.size(1) > 2:
            velocity = preds[:, 1:, :] - preds[:, :-1, :]
            acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]
            smoothness_loss = torch.norm(acceleration, dim=-1).mean()
            loss += self.smoothing_weight * smoothness_loss

        return loss

def compute_ade(preds, targets):
    return torch.norm(preds - targets, dim=-1).mean().item()

def compute_fde(preds, targets):
    return torch.norm(preds[:, -1, :] - targets[:, -1, :], dim=-1).mean().item()

def train(data_dir, epochs=1, scale_factor=50.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_massive = load_polars_dataframe(data_dir)
    if df_massive is None:
        return
    dataset = ArgoverseVehicleDataset(df_massive, past_steps=20, future_steps=30)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    
    model = TransformerMotionPredictor(d_model=768, nhead=8, num_layers=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = SmoothTrajectoryLoss(ade_weight=1.0, fde_weight=2.0, smoothing_weight=0.2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        steps_per_epoch=len(dataloader),
        epochs=epochs,
        pct_start=0.1
    )

    train_losses = []

    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        for batch_idx, (past_traj, target_future) in enumerate(dataloader):
            past_traj = past_traj.to(device) / scale_factor
            target_future = target_future.to(device) / scale_factor

            optimizer.zero_grad()
            predictions = model(past_traj)
            loss = criterion(predictions, target_future)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    return model
