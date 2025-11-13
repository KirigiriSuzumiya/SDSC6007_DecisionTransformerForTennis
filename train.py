import minari
from tqdm import tqdm
import numpy as np
from utils.datacollator import DecisionTransformerVisionDataCollator
from torchvision import transforms
from transformers import Trainer, TrainingArguments
from torchvision.models import MobileNet_V3_Large_Weights
from models.mobilenet_dt import MobileNetDT
from transformers import DecisionTransformerConfig


def load_data(dataset_id:str, episodes:int, act_dim:int, 
              img_preprocess)->DecisionTransformerVisionDataCollator:
    dataset = minari.load_dataset(dataset_id)
    print(f"env action space: {dataset.action_space}")

    print(f"sample {episodes} episodes from dataset {dataset_id}...")
    samples = dataset.sample_episodes(n_episodes=episodes)
    
    split_samples = []
    for sample in tqdm(samples, desc="split samples from episodes"):
        nozero_idxs = np.where(sample.rewards != 0)[0]
        start = 0
        for nonzero_idx in nozero_idxs:
            split_sample = {
                "score": sample.rewards[nonzero_idx],
                "rewards": sample.rewards[start:nonzero_idx+1],
                "actions": sample.actions[start:nonzero_idx+1],
                "observations": sample.observations[start:nonzero_idx+1],
                "dones": [0]*(nonzero_idx - start) + [1]
            }
            split_samples.append(split_sample)
            start = nonzero_idx + 1
    
    print(f"splite {len(split_samples)} samples")
    print(f"img preprocess: {img_preprocess}")
    collator = DecisionTransformerVisionDataCollator(
        split_samples, 
        act_dim=act_dim, 
        image_preprocess=img_preprocess
    )
    return collator, split_samples

def train(epochs, batchsize, model, split_samples, collator):
    training_args = TrainingArguments(
        output_dir="output/tennis/",
        remove_unused_columns=False,
        num_train_epochs=epochs,
        per_device_train_batch_size=batchsize,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        optim="adamw_torch",
        max_grad_norm=0.25,
        logging_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset= split_samples,
        data_collator=collator,
    )
    train_out = trainer.train()
    return train_out
    

if __name__ == "__main__":
    img_preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224]),
        MobileNet_V3_Large_Weights.IMAGENET1K_V2.transforms(),
    ])
    collator, split_samples = load_data('atari/tennis/expert-v11', 100, 18, img_preprocess)
    
    config = DecisionTransformerConfig(act_dim=18)
    model = MobileNetDT(config, MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    train(50, 16, model, split_samples, collator)
    