import torch
import pandas as pd
import plotly.express as px
from .train import compute_ade, compute_fde

def visualize_good_predictions(model, dataset, device, num_plots=10, combined_error_threshold=5.0, scale_factor=50.0):
    model.eval()
    plotted_count = 0
    total_plot_ade = 0.0
    total_plot_fde = 0.0

    print(f"Searching for {num_plots} predictions with (ADE + FDE) < {combined_error_threshold} meters...")

    color_discrete_map = {
        'Past Trajectory': '#00BFFF',
        'Ground Truth Future': '#32CD32',
        'Predicted Future': '#FF1493'
    }

    for idx in range(len(dataset)):
        if plotted_count >= num_plots:
            break

        seq = dataset.sequences[idx]

        x_raw = torch.tensor(seq['x'][:dataset.seq_len], dtype=torch.float32)
        y_raw = torch.tensor(seq['y'][:dataset.seq_len], dtype=torch.float32)
        features_abs = torch.stack([x_raw, y_raw], dim=1)

        current_pos = features_abs[dataset.past_steps - 1].clone()
        features_norm = features_abs - current_pos

        past_norm = features_norm[:dataset.past_steps].unsqueeze(0).to(device)
        future_norm = features_norm[dataset.past_steps:].unsqueeze(0).to(device)

        with torch.no_grad():
            pred_norm_scaled = model(past_norm / scale_factor)

        pred_norm = pred_norm_scaled * scale_factor

        ade = compute_ade(pred_norm, future_norm)
        fde = compute_fde(pred_norm, future_norm)
        total_error = ade + fde

        if total_error < combined_error_threshold:
            pred_abs = pred_norm.squeeze(0).cpu() + current_pos
            past_abs = features_abs[:dataset.past_steps]
            future_abs = features_abs[dataset.past_steps:]

            df_past = pd.DataFrame({'x': past_abs[:, 0].numpy(), 'y': past_abs[:, 1].numpy(), 'Type': 'Past Trajectory'})
            df_gt = pd.DataFrame({'x': future_abs[:, 0].numpy(), 'y': future_abs[:, 1].numpy(), 'Type': 'Ground Truth Future'})
            df_pred = pd.DataFrame({'x': pred_abs[:, 0].numpy(), 'y': pred_abs[:, 1].numpy(), 'Type': 'Predicted Future'})

            df_plot = pd.concat([df_past, df_gt, df_pred], ignore_index=True)

            fig = px.line(df_plot, x='x', y='y', color='Type', markers=True,
                          title=f"Track: {seq['track_id']} | ADE: {ade:.3f}m | FDE: {fde:.3f}m | Total: {total_error:.3f}m",
                          color_discrete_map=color_discrete_map)

            fig.update_layout(
                yaxis=dict(scaleanchor="x", scaleratio=1),
                plot_bgcolor='rgb(20, 20, 20)',
                paper_bgcolor='rgb(20, 20, 20)',
                font=dict(color='white'),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            fig.show()

            plotted_count += 1
            total_plot_ade += ade
            total_plot_fde += fde

    if plotted_count > 0:
        print(f"\nSuccessfully plotted {plotted_count} samples.")
        print(f"Average ADE for these displayed sequences: {total_plot_ade / plotted_count:.4f}m")
        print(f"Average FDE for these displayed sequences: {total_plot_fde / plotted_count:.4f}m")
    else:
        print("No sequences found below the error threshold.")
