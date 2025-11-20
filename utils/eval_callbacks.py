import os
from typing import Optional
import torch
from transformers import TrainerCallback

from eval import sample_dataset_using_model
from utils.preprocess import MobileNetDTPreprocessor
from models.vision_dt import vision_transforms


class EvalEveryNEpochsCallback(TrainerCallback):
    """Run `sample_dataset_using_model` every `n` epochs and log reward to trainer/TensorBoard.

    Args:
        n: run eval every n epochs (positive int)
        num_episodes: number of episodes to run per eval call
        act_dim: action dimension passed to preprocessor (default 18)
        stack_frame: frame stack size passed to preprocessor (default 4)
        max_len: max sequence length for preprocessor (default 100)
    """

    def __init__(
        self,
        trainer,
        n: int = 10,
        num_episodes: int = 1,
        act_dim: int = 18,
        stack_frame: int = 4,
        max_len: int = 100,
        max_step: int = 10,
    ):
        if n <= 0:
            raise ValueError("n must be > 0")
        self.n = int(n)
        self.num_episodes = int(num_episodes)
        self.act_dim = act_dim
        self.stack_frame = stack_frame
        self.max_len = max_len
        self.trainer = trainer
        self.max_step = max_step
        # keep track of the best metric seen so far (higher is better)
        self.best_metric = None
        self.best_dir = None

    def on_epoch_end(self, args, state, control, model, **kwargs):
        if state.epoch is None:
            return control

        current_epoch = int(round(state.epoch))
        if current_epoch <= 0 or (current_epoch % self.n) != 0:
            return control

        model.eval()

        preprocessor = MobileNetDTPreprocessor(
            mode="eval",
            act_dim=self.act_dim,
            img_preprocess=vision_transforms,
            stack_frame=self.stack_frame,
            max_len=self.max_len,
        )

        save_path = os.path.join(args.output_dir, "record", f"epoch-{current_epoch}")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        # Call eval function without saving dataset
        reward_out = sample_dataset_using_model(
            save_path=save_path,
            model=model,
            num_episodes=self.num_episodes,
            device=device,
            preprocessor=preprocessor,
            description=f"eval-epoch-{current_epoch}",
            dataset_id=None,
            save_video=True,
            max_step=self.max_step,
        )
        # Log the metric to the trainer (so it appears in TensorBoard / logs)
        self.trainer.log({"win_rate": reward_out})

        # Save model if this is the best metric so far
        try:
            # reward_out expected to be a numeric scalar (higher is better)
            current_metric = float(reward_out)
        except Exception:
            # if metric cannot be interpreted as float, skip saving
            return control

        improved = False
        if self.best_metric is None or current_metric > self.best_metric:
            improved = True

        if improved:
            self.best_metric = current_metric
            # save to a stable "best" directory under the trainer output dir
            best_dir = os.path.join(args.output_dir, "best")
            os.makedirs(best_dir, exist_ok=True)

            # trainer.save_model will save the model weights/config to the directory
            try:
                self.trainer.save_model(best_dir)
                # also update trainer state so external tools can find the best checkpoint
                try:
                    self.trainer.state.best_model_checkpoint = best_dir
                except Exception:
                    # not critical
                    pass
                # record the path for reference
                self.best_dir = best_dir
                # log the save event
                self.trainer.log({"best_model_saved_at": best_dir, "best_win_rate": self.best_metric})
            except Exception as e:
                # don't crash training because of save failure; just log
                try:
                    self.trainer.log({"best_model_save_error": str(e)})
                except Exception:
                    pass
