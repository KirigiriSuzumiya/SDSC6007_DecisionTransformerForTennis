import torch

def frame_stack(frames, stack_size=4, fill_zero=False):
    T = len(frames)
    C, H, W = frames[0].shape
    stacked = torch.zeros((T, stack_size, H, W), dtype=frames[0].dtype, device=frames[0].device)
    for t in range(T):
        for k in range(stack_size):
            idx = t - (stack_size - 1) + k
            if idx < 0:
                if fill_zero:
                    stacked[t, k] = 0
                else:
                    stacked[t, k] = frames[0][0]
            else:
                stacked[t, k] = frames[idx][0]
    return stacked