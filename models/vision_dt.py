from transformers import DecisionTransformerConfig, DecisionTransformerModel
from torchvision import models
from torch import nn
import torch
from torchvision import transforms

vision_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]
)
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        inputs = x
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, channels, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(channels)
        self.res2 = ResidualBlock(channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class VisualEncoder(nn.Module):
    def __init__(self, input_channels=4, output_dim=512):
        super().__init__()
        self.seq1 = ConvSequence(input_channels, 16)
        self.seq2 = ConvSequence(16, 32)
        self.seq3 = ConvSequence(32, 32)
        # self.pool = nn.AdaptiveAvgPool2d((5, 5))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(17280, 4096)
        self.fc2 = nn.Linear(4096, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        # x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

class VisualDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)
        # replace the state embedding with a vision model
        self.embed_state = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=1),
            VisualEncoder(output_dim=config.hidden_size),
            nn.Unflatten(dim=0, unflattened_size=(-1, 100)),
        )

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        loss = torch.nn.CrossEntropyLoss()(action_preds, torch.argmax(action_targets, dim=1))
        latest_action = torch.argmax(action_preds[-1])

        return {"loss": loss, "current_action":latest_action}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)