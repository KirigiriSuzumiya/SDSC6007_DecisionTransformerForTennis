import minari
from tqdm import tqdm
import numpy as np
from utils.datacollator import DecisionTransformerVisionDataCollator
from utils.framestack import frame_stack
from utils.sample_by_rate import sample_by_ratio
from torchvision import transforms
from transformers import Trainer, TrainingArguments
from torchvision.models import MobileNet_V3_Large_Weights
from models.mobilenet_dt import MobileNetDT, mobilenet_transforms
from models.vision_dt import VisualDT, vision_transforms
from transformers import DecisionTransformerConfig


def load_data(dataset_id:str, episodes:int):
    dataset = minari.load_dataset(dataset_id)
    print(f"env action space: {dataset.action_space}")

    print(f"sample {episodes} episodes from dataset {dataset_id}...")
    episodes = dataset.sample_episodes(n_episodes=episodes)
    
    samples = []
    for episode in tqdm(episodes, desc="origanize episodes"):
        nozero_idxs = np.where(episode.rewards != 0)[0]
        start = 0
        for end in nozero_idxs:
            sample = {
                "score": episode.rewards[end],
                "rewards": episode.rewards[start:end+1],
                "actions": episode.actions[start:end+1],
                "observations": episode.observations[start:end+1],
                "dones": [0]*(len(episode.actions[start:end+1])-1) + [1]
            }
            samples.append(sample)
    
    samples = sample_by_ratio(samples)
    return samples

def train(epochs, batchsize, model, samples, collator):
    training_args = TrainingArguments(
        output_dir="output/tennis/",
        remove_unused_columns=False,
        num_train_epochs=epochs,
        save_strategy="epoch",
        save_total_limit=2,
        per_device_train_batch_size=batchsize,
        learning_rate= 1e-3,
        weight_decay=0.1,
        warmup_ratio=0.1,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=0.25,
        logging_dir="output/logs", # Directory for TensorBoard logs
        logging_steps=10, # Log every 10 steps
        report_to=["tensorboard"],
    )

    from utils.eval_callbacks import EvalEveryNEpochsCallback

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset= samples,
        data_collator=collator,
    )
    eval_callback = EvalEveryNEpochsCallback(trainer, n=1, max_step=2000)
    trainer.add_callback(eval_callback)
    train_out = trainer.train()
    return train_out
    

if __name__ == "__main__":
    samples = load_data('atari/tennis/expert-v11', 100)
    
    collator = DecisionTransformerVisionDataCollator(
        samples, 
        act_dim=18, 
        stack_frame=4,
        img_preprocess=vision_transforms
    )
    
    config = DecisionTransformerConfig(
        act_dim=18,
        n_head=16,
        n_layer=12,
        hidden_size=512,
    )
    model = VisualDT(config)
    train(100, 16, model, samples, collator)
    