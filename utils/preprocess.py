import random
import numpy as np
from torchvision import transforms
from torchvision.models import MobileNet_V3_Large_Weights

class MobileNetDTPreprocessor():
    def __init__(
        self, 
        mode:str, 
        act_dim:int,
        max_len:int=50,
        pretrain_weights:MobileNet_V3_Large_Weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        max_ep_len: int = 1000, # max episode length in the dataset
        scale: float = 1000.0  # normalization of rewards/returns
    ): # mode: train/eval
        self.mode = mode
        self.max_len = max_len
        self.act_dim = act_dim
        self.img_preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([224]),
            pretrain_weights.transforms(),
        ])
        self.max_ep_len = max_ep_len
        self.scale = scale
        pass
    def __call__(self, observations, rewards, actions, dones):
        # a batch of dataset features
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        if self.mode == "train":
            si = random.randint(0, len(observations) - 1)
        elif self.mode == "eval":
            si = max(len(observations) - self.max_len, 0)
        else:
            raise f"invalid mode {self.mode}"
        # get sequences from dataset
        # TODO: replace with image observations
        obs_img = [self.img_preprocess(obs) for obs in observations[si : si + self.max_len]]
        s.append(np.array(obs_img).reshape(1, -1, 3, obs_img[0].shape[1], obs_img[0].shape[2]))  # reshape for image observations
        one_hot_actions = np.eye(self.act_dim)[actions[si : si + self.max_len]].reshape(1, -1, self.act_dim)
        a.append(one_hot_actions)
        r.append(np.array(rewards[si : si + self.max_len]).reshape(1, -1, 1))

        d.append(np.array(dones[si : si + self.max_len]).reshape(1, -1))
        timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
        timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
        rtg.append(
            self._discount_cumsum(np.array(rewards[si:]), gamma=1.0)[
                : s[-1].shape[1]   # TODO check the +1 removed here
            ].reshape(1, -1, 1)
        )
        if rtg[-1].shape[1] < s[-1].shape[1]:
            print("if true")
            rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
            
        # padding and state + reward normalization
        tlen = s[-1].shape[1]
        s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 3, s[-1].shape[3], s[-1].shape[4])), s[-1]], axis=1)
        # s[-1] = (s[-1] - self.state_mean) / self.state_std
        a[-1] = np.concatenate(
            [np.ones((1, self.max_len - tlen, self.act_dim)) * -10.0, a[-1]],
            axis=1,
        )
        r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
        d[-1] = np.concatenate([np.ones((1, self.max_len - tlen)) * 2, d[-1]], axis=1)
        rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
        timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
        mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))
        return s, a, r, d, rtg, timesteps, mask
    
    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum