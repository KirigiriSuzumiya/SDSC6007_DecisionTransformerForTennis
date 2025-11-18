from transformers import DecisionTransformerConfig, DecisionTransformerModel
from torchvision import models
from torch import nn
import torch
from torchvision import transforms
from torchvision.models import MobileNet_V3_Large_Weights

mobilenet_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([224]),
    MobileNet_V3_Large_Weights.IMAGENET1K_V2.transforms(),
])

class MobileNetDT(DecisionTransformerModel):
    def __init__(self, config, pretrain_weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2):
        super().__init__(config)
        # replace the state embedding with a vision model
        pretrained_mobilenet = models.mobilenet_v3_large(weights=pretrain_weights)
        mobilenet_feature_size = pretrained_mobilenet.classifier[0].in_features
        pretrained_mobilenet.classifier = nn.Linear(mobilenet_feature_size, config.hidden_size)
        self.embed_state = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=1),
            pretrained_mobilenet,
            nn.Unflatten(dim=0, unflattened_size=(-1, 50)),
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