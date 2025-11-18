from pettingzoo.atari import tennis_v3
import pickle
import supersuit
from models.mobilenet_dt import MobileNetDT
import torch
import os
from tqdm import tqdm
import numpy as np
from utils.preprocess import MobileNetDTPreprocessor
from utils.swap_color import swap_colors
from utils.save_image import save_numpy_frames_to_video
from models.vision_dt import VisualDT,vision_transforms
from transformers import AutoConfig

def extract_envstep_by_agents(agents_name, observation, reward):
    if agents_name == "first_0":
        return observation[agents_name], reward[agents_name]
    elif agents_name == "second_0":
        return swap_colors(observation[agents_name], [117, 128, 240], [240, 128, 128]), reward[agents_name]
    return 

def sample_dataset_using_model(
    save_path, 
    model:MobileNetDT, 
    num_episodes, 
    device,
    preprocessor:MobileNetDTPreprocessor,
    save_video:bool=True,
    agent_ids = ["first_0", "second_0"],
    return_dataset:bool=False ,
    max_step:int=10*1000,
):
    env = tennis_v3.parallel_env()
    env = supersuit.frame_skip_v0(env, 4)
    env.reset()
    
    episodes_data = []
    for episode_idx in tqdm(range(num_episodes), desc="Generating episodes"):
        episode_steps = [] 
        episode_total_reward = 0
        env.reset()
        done = False
        
        video_frame = []
        
        observation, reward, terminated, truncated, info = env.step({
            "first_0": 0,
            "second_0": 0,
        })
        
        observations = {}
        rewards = {}
        reward_out = {}
        actions = {}
        dones = {}
        for agent in agent_ids:
            obs_agent, rew_agent = extract_envstep_by_agents(agent, observation, reward)
            observations[agent] = [obs_agent]
            rewards[agent] = [rew_agent]
            reward_out[agent] = 0
            actions[agent] = [0]
            dones[agent] = [False]
            
        step_cnt = 0
        while not done:
            mutli_agent_action = {}
            for agent in agent_ids:
                s, a, r, d, rtg, timesteps, mask = preprocessor(observations[agent], rewards[agent], actions[agent], dones[agent])
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
                mutli_agent_action[agent] = int(out["current_action"].cpu())
                
            obs, rew, terminated, truncated, info = env.step(mutli_agent_action)
            step_cnt +=1
            
            if save_video:
                video_frame.append(obs[agent_ids[0]])
            if return_dataset:
                episode_steps.append({
                    "observation": obs,
                    "reward":rew,
                    "action":mutli_agent_action,
                    "terminated":terminated,
                    "truncated":truncated,
                    "info":info
                })
                
            for agent in agent_ids:
                obs_agent, rew_agent = extract_envstep_by_agents(agent, obs, rew)
                if int(rew[agent]) != 0:
                    reward_out[agent] += int(rew[agent])
                    observations[agent] = []
                    rewards[agent] = []
                    actions[agent] = []
                    dones[agent] = []
                done = terminated[agent] or truncated[agent] or step_cnt >= max_step
                observations[agent].append(obs_agent)
                rewards[agent].append(rew_agent)
                actions[agent].append(mutli_agent_action[agent])
                dones[agent].append(done)
            
        if return_dataset:
            episodes_data.append({
                "episode_id": episode_idx,
                "steps": episode_steps,
            })
        if save_video:
            video_path = os.path.join(save_path, f"mutliagent-episode-{episode_idx}.mp4")
            print(f"saving video to {video_path}")
            save_numpy_frames_to_video(
                video_frame,
                video_path
            )

    # if dataset_path:
    #     pkl_path = os.path.join(dataset_path, f"episode-{num_episodes}")
    #     print(f"saving dataset to {pkl_path}")
    #     pickle.dump(episodes_data, open(pkl_path, "wb"))
    env.close()
    if return_dataset:
        return episodes_data
    else:
        return reward_out,

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
        device=device,
        save_video=True,
        return_dataset=True,
    )
    print(out)
    