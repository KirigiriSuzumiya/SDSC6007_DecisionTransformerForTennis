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
    episodes = dataset.sample_episodes(n_episodes=episodes)
    
    samples = []
    for episode in tqdm(episodes, desc="origanize episodes"):
        for start in range(0, len(episode.actions), 1000):
            end = min(start+1000, len(episode.actions)-1)
            sample = {
                "score": np.sum(episode.rewards[start:end]),
                "rewards": episode.rewards[start:end],
                "actions": episode.actions[start:end],
                "observations": episode.observations[start:end],
                "dones": [0]*(len(episode.actions[start:end])-1) + [1]
            }
            samples.append(sample)
    
    collator = DecisionTransformerVisionDataCollator(
        samples, 
        act_dim=act_dim, 
    )
    return collator, np.random.rand(len(samples)*10, 1)

def train(epochs, batchsize, model, samples, collator):
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
        train_dataset= samples,
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
    collator, samples = load_data('atari/tennis/expert-v11', 100, 18, img_preprocess)
    
    config = DecisionTransformerConfig(act_dim=18)
    model = MobileNetDT(config, MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    train(50, 8, model, samples, collator)
    