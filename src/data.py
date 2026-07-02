import s3fs
import os
from tqdm.auto import tqdm
import concurrent.futures
import polars as pl
import pyarrow.dataset as ds
import torch
from torch.utils.data import Dataset, DataLoader

# Argoverse 2 public S3 bucket path
S3_BASE_PATH = "argoverse/datasets/av2/motion-forecasting"

def download_av2_split(split, num_scenarios, download_dir):
    fs = s3fs.S3FileSystem(anon=True)
    s3_split_path = f"{S3_BASE_PATH}/{split}"
    print(f"Fetching scenario list from {s3_split_path}...")

    try:
        scenarios = fs.ls(s3_split_path)[:num_scenarios]
    except Exception as e:
        print(f"Error listing S3 path: {e}")
        return

    split_dir = os.path.join(download_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    print(f"Starting download of {len(scenarios)} {split} scenarios...")

    def download_scenario(scenario_s3_path):
        try:
            files = fs.ls(scenario_s3_path)
            for f in files:
                if f.endswith('.parquet'):
                    local_path = os.path.join(split_dir, os.path.basename(f))
                    if not os.path.exists(local_path):
                        fs.get(f, local_path)
        except Exception as e:
            pass 

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        list(tqdm(executor.map(download_scenario, scenarios), total=len(scenarios), desc=f"Downloading {split}"))


def load_polars_dataframe(data_dir):
    try:
        arrow_dataset = ds.dataset(data_dir, format="parquet")
        df_lazy = pl.scan_pyarrow_dataset(arrow_dataset)
        df_vehicles_lazy = df_lazy.filter(pl.col("object_type") == "vehicle") \
                                  .select(["scenario_id", "track_id", "timestep", "position_x", "position_y"]) \
                                  .with_columns([
                                      pl.col("timestep").cast(pl.Int32),
                                      pl.col("position_x").cast(pl.Float32),
                                      pl.col("position_y").cast(pl.Float32)
                                  ])
        print("Collecting data into memory...")
        df_massive = df_vehicles_lazy.collect()
        print(f"Successfully loaded dataframe with {df_massive.height} rows.")
        return df_massive
    except Exception as e:
        print(f"Could not load from {data_dir}. Error: {e}")
        return None


class ArgoverseVehicleDataset(Dataset):
    def __init__(self, df: pl.DataFrame, past_steps=20, future_steps=30):
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.seq_len = past_steps + future_steps

        print("Grouping data into sequences by scenario and track ID...")
        grouped_df = df.group_by(["scenario_id", "track_id"]).agg([
            pl.col("position_x").alias("x"),
            pl.col("position_y").alias("y")
        ])
        
        valid_sequences = grouped_df.filter(pl.col("x").list.len() >= self.seq_len)
        self.sequences = valid_sequences.to_dicts()
        print(f"Extracted {len(self.sequences)} valid full sequences.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        x_tensor = torch.tensor(seq['x'][:self.seq_len], dtype=torch.float32)
        y_tensor = torch.tensor(seq['y'][:self.seq_len], dtype=torch.float32)

        features = torch.stack([x_tensor, y_tensor], dim=1)

        # Normalization Fix
        current_pos = features[self.past_steps - 1].clone()
        features = features - current_pos

        past_traj = features[:self.past_steps]
        future_traj = features[self.past_steps:]

        return past_traj, future_traj
