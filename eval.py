from models.mobilenet_dt import MobileNetDT
import torch
import minari
from minari import DataCollector
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from tqdm import tqdm
import numpy as np
from utils.preprocess import MobileNetDTPreprocessor
from models.vision_dt import VisualDT,vision_transforms
from transformers import AutoConfig
def sample_dataset_using_model(
    save_path, 
    model:MobileNetDT, 
    num_episodes, 
    device,
    preprocessor:MobileNetDTPreprocessor,
    description:str="MobileNetDT",
    dataset_id:str=None,
    save_video:bool=True,
    max_step:int=10**100,
):
    ori_dataset = minari.load_dataset('atari/tennis/expert-v11')
    env = ori_dataset.recover_environment(
        eval_env=True,
        render_mode="rgb_array",
    )
    if save_video:
        env = RecordVideo(
            env,
            video_folder=save_path,    # Folder to save videos
            name_prefix="singleagent",               # Prefix for video filenames
            episode_trigger=lambda x: True    # Record every episode
        )
    if dataset_id:
        env = DataCollector(env, record_infos=False)
    
    total_steps = 0
    for _ in tqdm(range(num_episodes), desc="Generating single agent episodes"):
        episodic_return = 0
        env.reset()
        done = False
        
        obseration, reward, terminated, truncated, info = env.step(0)
        observations = [obseration]
        rewards = [reward]
        reward_out = {
            "win": 0,
            "loss": 0,
        }
        actions = [0]
        dones = [False]
        while not done:
            s, a, r, d, rtg, timesteps, mask = preprocessor(observations, rewards, actions, dones)
            s = torch.from_numpy(np.concatenate(s, axis=0)).float().to(device)
            a = torch.from_numpy(np.concatenate(a, axis=0)).float().to(device)
            r = torch.from_numpy(np.concatenate(r, axis=0)).float().to(device)
            d = torch.from_numpy(np.concatenate(d, axis=0)).to(device)
            rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float().to(device)
            timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long().to(device)
            mask = torch.from_numpy(np.concatenate(mask, axis=0)).float().to(device)
            with torch.no_grad():
                out= model(
                    states=s,
                    actions=a,
                    rewards=r,
                    returns_to_go=rtg,
                    timesteps=timesteps,
                    attention_mask=mask
                )
            action = int(out["current_action"].cpu())
            
            obs, rew, terminated, truncated, info = env.step(action)
            total_steps += 1
            if int(rew) != 0:
                if int(rew) > 0:
                    reward_out["win"] += 1
                else:
                    reward_out["loss"] += 1
                observations = []
                rewards = []
                actions = []
                dones = []
            done = terminated or truncated or total_steps >= max_step
            observations.append(obs)
            rewards.append(rew)
            actions.append(action)
            dones.append(done)

    if dataset_id:
        dataset = env.create_dataset(
            dataset_id=dataset_id,
            author="Yifan BU",
            author_email="boyifan1@126.com",
            algorithm_name="MobileNetDT",
            description=description,
            code_permalink="https://github.com/KirigiriSuzumiya/SDSC6007_DecisionTransformerForTennis",
            requirements=["gymnasium[atari,accept-rom-license]"]
        )
    env.close()
    return reward_out["win"] / (reward_out["loss"] + reward_out["win"] + 1e-5)


if __name__ == "__main__":
    save_path = "./output/record/checkpoint-22400"
    model_path = "/root/SDSC6007_DecisionTransformerForTennis/output/tennis/checkpoint-22400"
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    config = AutoConfig.from_pretrained(model_path)
    model = VisualDT.from_pretrained(model_path, config=config)
    model.to(device)
    model.eval()
    
    
    
    preprocessor = MobileNetDTPreprocessor(
        mode='eval',
        act_dim=18,
        img_preprocess=vision_transforms,
        stack_frame=4,
        max_len=100,
    )
    
    out = sample_dataset_using_model(
        save_path=save_path,
        model=model,
        num_episodes=1,
        preprocessor=preprocessor,
        description="mobilenet dt checkpoint-13650",
        # dataset_id="atari/tennis/naive-v0",
        device=device,
    )
    print(out)
    