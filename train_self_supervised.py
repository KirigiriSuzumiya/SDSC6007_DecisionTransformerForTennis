import minari
from tqdm import tqdm
import numpy as np
from utils.datacollator import DecisionTransformerVisionDataCollator
from utils.framestack import frame_stack
from utils.sample_by_rate import sample_by_ratio
from transformers import Trainer, TrainingArguments
from transformers import AutoConfig
from models.vision_dt import VisualDT, vision_transforms
from mutli_agent_eval import sample_dataset_using_model, extract_envstep_by_agents
from tqdm import tqdm
import numpy as np
from utils.preprocess import MobileNetDTPreprocessor
from utils.swap_color import swap_colors
from models.vision_dt import VisualDT,vision_transforms
import torch


def load_data(dataset, act_dim:int, 
              img_preprocess,
              stack_frame:int=None
)->DecisionTransformerVisionDataCollator:
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
    collator = DecisionTransformerVisionDataCollator(
        samples, 
        act_dim=act_dim, 
        stack_frame=stack_frame,
        img_preprocess=img_preprocess
    )
    return collator, np.random.rand(len(samples)*5, 1)

def train(epochs, batchsize, model, samples, collator):
    training_args = TrainingArguments(
        output_dir="output/self_supervised_tennis/",
        remove_unused_columns=False,
        num_train_epochs=epochs,
        per_device_train_batch_size=batchsize,
        learning_rate= 6 * 1e-4,
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
    save_path:str, 
    device,
    sample_episodes
):
    model.eval()
    eval_preprocessor = MobileNetDTPreprocessor(
        mode='eval',
        act_dim=18,
        img_preprocess=vision_transforms,
        stack_frame=4,
        max_len=100,
    )
    
    episodes_data = sample_dataset_using_model(
        save_path=save_path,
        model=model,
        num_episodes=sample_episodes,
        preprocessor=eval_preprocessor,
        device=device,
        save_video=True,
        return_dataset=True,
    )
    # TODO
    collactor, samples = load_data(
        episodes_data,
        18,
        vision_transforms,
        stack_frame=4
    )
    train(10, 16, model, samples, collactor)
    
    
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
    for round in range(self_supervised_round):
        print(f"self_supervised_round {round}:")
        self_supervised_train(
            model, 
            save_path="/root/SDSC6007_DecisionTransformerForTennis/output/record/self_supervised-training",
            device=device,
            sample_episodes=sample_episodes
        )
    