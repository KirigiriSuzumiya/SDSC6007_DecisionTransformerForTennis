from dataclasses import dataclass
import random
import torch
import numpy as np
from .preprocess import MobileNetDTPreprocessor

@dataclass
class DecisionTransformerVisionDataCollator:
    return_tensors: str = "pt"
    max_len: int = 100 #subsets of the episode we use for training
    # state_dim: int = 17  # size of state space
    act_dim: int = 18  # size of action space
    max_ep_len: int = 4096 # max episode length in the dataset
    scale: float = 1000.0  # normalization of rewards/returns
    state_mean: np.array = None  # to store state means
    state_std: np.array = None  # to store state stds
    p_sample: np.array = None  # a distribution to take account trajectory lengths
    n_traj: int = 0 # to store the number of trajectories in the dataset

    def __init__(self, dataset, act_dim, img_preprocess, stack_frame:int=None) -> None:
        self.act_dim = act_dim
        # self.state_dim = len(dataset[0]["observations"][0])
        self.dataset = dataset
        # calculate dataset stats for normalization of states
        # states = []
        traj_lens = []
        for obs in dataset:
            # states.extend(obs["observations"])
            traj_lens.append(len(obs["observations"]))
        self.n_traj = len(traj_lens)
        
        self.preprocessor = MobileNetDTPreprocessor(
            mode="train",
            act_dim=act_dim,
            max_len=self.max_len,
            max_ep_len=self.max_ep_len,
            scale=self.scale,
            stack_frame=stack_frame,
            img_preprocess=img_preprocess
        )
        
        traj_lens = np.array(traj_lens)
        self.p_sample = traj_lens / sum(traj_lens)

    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        batch_size = len(features)
        # this is a bit of a hack to be able to sample of a non-uniform distribution
        batch_inds = np.random.choice(
            np.arange(self.n_traj),
            size=batch_size,
            replace=True,
            p=self.p_sample,  # reweights so we sample according to timesteps
        )
        # a batch of dataset features
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        
        for ind in batch_inds:
            # for feature in features:
            feature = self.dataset[int(ind)]

            s_curr, a_curr, r_curr, d_curr, rtg_curr, timesteps_curr, mask_curr = self.preprocessor(
                feature["observations"],
                feature["rewards"],
                feature["actions"],
                feature["dones"]
            )
            s.extend(s_curr)
            a.extend(a_curr)
            r.extend(r_curr)
            d.extend(d_curr)
            rtg.extend(rtg_curr)
            timesteps.extend(timesteps_curr)
            mask.extend(mask_curr)

        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        d = torch.from_numpy(np.concatenate(d, axis=0))
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()

        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }