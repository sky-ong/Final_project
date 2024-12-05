import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
from tqdm import tqdm
from base_experiment import BaseExperiment
from spatial import JumpCNF, SelfAttentiveCNF, ConditionalGMM, IndependentCNF, IndependentNF, SelfAttentiveNF
from temporal import NeuralPointProcess
from datasets import spatiotemporal_events_collate_fn as collate_fn

class CovidNJDataset(Dataset):
    def __init__(self, data_dir, split, max_sequences=None):
        """
        Args:
            data_dir (str): Path to the directory containing the CSV files.
            split (str): One of 'train', 'val', or 'test'.
            max_sequences (int): Maximum number of sequences to load for this split.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_sequences = max_sequences
        self.files = sorted(self.data_dir.glob("*.csv"))
        
        if split == "train":
            self.files = self.files[:384]
        elif split == "val":
            self.files = self.files[384:466]
        elif split == "test":
            self.files = self.files[466:549]
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'.")
        
        if max_sequences:
            self.files = self.files[:max_sequences]
        
        self.data = []
        for file in tqdm(self.files, desc=f"Loading {split} data"):
            df = pd.read_csv(file)
            self.data.append(df)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        df = self.data[idx]
        time = torch.tensor(df["time"].values, dtype=torch.float32)
        spatial = torch.tensor(df[["X_noisy", "Y_noisy"]].values, dtype=torch.float32)
        mask = torch.ones(len(df), dtype=torch.float32)  # All data points are valid
        return time, spatial, mask


class STPP(BaseExperiment):
    def get_model(self, args):
        hidden_dims = [args.hidden_dim] * args.hidden_layers
        if args.model == 'ode':
            if args.density_model == 'independent':
                return IndependentCNF(self.dim, hidden_dims)
            elif args.density_model == 'attention':
                return SelfAttentiveCNF(self.dim, hidden_dims)
        elif args.model == 'flow':
            if args.density_model == 'independent':
                return IndependentNF(self.dim, hidden_dims, n_layers=args.flow_layers,
                                    time_net=args.time_net, time_hidden_dim=args.time_hidden_dim, device=self.device)
            elif args.density_model == 'attention':
                return SelfAttentiveNF(self.dim, hidden_dims, n_layers=args.flow_layers,
                                    time_net=args.time_net, time_hidden_dim=args.time_hidden_dim, device=self.device)
        else:
            raise ValueError(f"Unsupported model type: {args.model}")


    def get_data(self, args):
        if args.data == 'covid':
            dataset_cls = CovidNJDataset
        else:
            raise NotImplementedError

        # assuming all times are normalized to (0,1) interval
        self.t0, self.t1 = torch.tensor([0.0]).to(self.device), torch.tensor([1.0]).to(self.device)

        def get_dl(split):
            dataset = dataset_cls(data_dir="output_intervals_8hours_overlap", split=split)
            return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        dltrain = get_dl('train')
        dlval = get_dl('val')
        dltest = get_dl('test')
        return 2, None, dltrain, dlval, dltest

    def _get_loss(self, batch):
        t, x, m = (s.to(self.device) for s in batch)
        likelihood = self.model.logprob(t, x, m)
        loss = -(likelihood * m).sum() / m.sum()
        return loss

    def training_step(self, batch):
        return self._get_loss(batch)

    def _get_loss_for_dl(self, dl):
        losses = []
        for batch in dl:
            losses.append(self._get_loss(batch).item())
        return np.mean(losses)

    def validation_step(self):
        return self._get_loss_for_dl(self.dlval)

    def test_step(self):
        return self._get_loss_for_dl(self.dltest)

    def finish(self):
        OUT_DIR = Path('/opt/ml/model')
        if OUT_DIR.exists():
            torch.save(self.model.state_dict(), OUT_DIR / 'model.pt')