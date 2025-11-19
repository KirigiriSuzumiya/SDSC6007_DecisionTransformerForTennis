import torch
from tqdm import tqdm
from utils.datacollator import DecisionTransformerVisionDataCollator
from utils.sample_by_rate import sample_by_ratio
from transformers import Trainer, TrainingArguments
from transformers import AutoConfig
import random
from tqdm import tqdm
import os

from utils.preprocess import MobileNetDTPreprocessor
from utils.swap_color import swap_colors
from models.vision_dt import VisualDT,vision_transforms
from models.vision_dt import VisualDT, vision_transforms
from mutli_agent_eval import sample_mutli_agent_dataset_using_model, extract_envstep_by_agents
from eval import sample_dataset_using_model
from train import load_data


def load_multi_agent_data(dataset):
    samples = []
    for episode in tqdm(dataset, desc="origanize episodes"):
        curr = 0
        sample = {
            "first_0":{
                "score": 0,
                "rewards": [],
                "actions": [],
                "observations": [],
                "dones": []
            },
            "second_0":{
                "score": 0,
                "rewards": [],
                "actions": [],
                "observations": [],
                "dones": []
            }
        }
        for step in episode["steps"]:
            for agent in ["first_0", "second_0"]:
                observation, reward = extract_envstep_by_agents(agent, step["observation"], step["reward"])
                sample[agent]["rewards"].append(int(reward))
                sample[agent]["observations"].append(observation)
                sample[agent]["actions"].append(int(step["action"][agent]))
                
                if int(step["reward"][agent])  != 0:
                    sample[agent]["score"] = int(step["reward"][agent])
                    sample[agent]["dones"].append(1)
                    samples.append(sample[agent])
                    sample[agent] = {
                        "score": 0,
                        "rewards": [],
                        "actions": [],
                        "observations": [],
                        "dones": []
                    }
                else:
                    sample[agent]["dones"].append(0)
                    pass
    
    samples = sample_by_ratio(samples)
    
    return samples

def train(epochs, batchsize, model, samples, collator, output_dir="output/self_supervised_tennis/"):
    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        num_train_epochs=epochs,
        per_device_train_batch_size=batchsize,
        learning_rate= 1e-4,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset= samples,
        data_collator=collator,
    )
    train_out = trainer.train()
    return train_out
    
def self_supervised_train(
    model:VisualDT, 
    device,
    sample_episodes,
    num_rounds:int=10,
):
    model.eval()
    eval_preprocessor = MobileNetDTPreprocessor(
        mode='eval',
        act_dim=18,
        img_preprocess=vision_transforms,
        stack_frame=4,
        max_len=100,
    )
    for round in range(num_rounds):
        print(f"sampling round {round}...")
        video_save_path = os.path.join("output/record", f"round-{round}")
        model_save_path = os.path.join("output/self_supervised_tennis/", f"round-{round}")
        
        # sample agent dataset
        dataset_id = f"atari/tennis/selfsupervised-v{round}"
        sample_dataset_using_model(
            video_save_path, 
            model=model, 
            num_episodes=sample_episodes//5*3, 
            device=device,
            preprocessor=eval_preprocessor,
            dataset_id=dataset_id,
            save_video=True,
        )
        single_agent_samples = load_data(dataset_id, sample_episodes//2)
        
        # sample multi-agent dataset
        episodes_data = sample_mutli_agent_dataset_using_model(
            save_path=video_save_path,
            model=model,
            num_episodes=sample_episodes//5*2,
            preprocessor=eval_preprocessor,
            device=device,
            save_video=True,
            return_dataset=True,
        )
        mutli_agent_samples = load_multi_agent_data(episodes_data)
        
        print(f"{len(mutli_agent_samples)} samples from mutli-agent sampling, {len(single_agent_samples)} samples from single-agent sampling.")
        samples = mutli_agent_samples + single_agent_samples
        random.shuffle(samples)
        
        collactor = DecisionTransformerVisionDataCollator(
            samples, 
            act_dim=18, 
            stack_frame=4,
            img_preprocess=vision_transforms
        )
        
        train(30, 16, model, samples, collactor, model_save_path)
    
    
if __name__ == "__main__":
    model_path = "/root/SDSC6007_DecisionTransformerForTennis/output/tennis/checkpoint-22400"
    self_supervised_round = 5
    sample_episodes = 10
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    config = AutoConfig.from_pretrained(model_path)
    model = VisualDT.from_pretrained(model_path, config=config)
    model.to(device)
    
    self_supervised_train(
        model,
        device=device,
        sample_episodes=sample_episodes,
        num_rounds=self_supervised_round,
    )
    