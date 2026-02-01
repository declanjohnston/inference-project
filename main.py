from mlh.hypers import Hypers, TBD
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.model import get_base_model, get_tokenizer
from dataclasses import dataclass, asdict
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from src.dataloaders import get_train_test_datasets
import wandb
from pathlib import Path
@dataclass
class Args(Hypers):
    
    # Project configuration
    project_name: str = "inference-project-final"
    hub_model_id: str = "djohnston5/gemma-2-2b-sft"
    save_model_weights: bool = False
    save_local: bool = True
    
    # Model and dataset configuration
    model_name: str = "google/gemma-2-2b"
    dataset_name: str = "tatsu-lab/alpaca"
    split_marker: str = "### Response:\n"
    # Placeholders for training objects
    model: AutoModelForCausalLM = TBD()
    tokenizer: AutoTokenizer = TBD()
    train: Dataset = TBD()
    test: Dataset = TBD()
    trainer: SFTTrainer = TBD()
    config: SFTConfig = TBD()
    
    # Hyperparameters
    learning_rate: float = 5e-6
    batch_size: int = 4
    gradient_accumulation_steps: int = 1  # 1 = off (no accumulation)
    weight_decay: float = 0.0  # 0 = off
    max_grad_norm: float = 1.0  # gradient clipping
    max_seq_length: int = 1024
    max_length: int = 1024
    output_dir: str = "./sft_output"
    max_steps: int = 12000
    logging_steps: int = 10
    eval_steps: int = 500
    
    
def init(args: Args):
    wandb.init(
        entity="declanjohnston5-declan-johnston",
        project=args.project_name,
        config=args.to_dict(),
    )
    
    args.model = get_base_model(args)
    args.tokenizer = get_tokenizer(args)
    args.train, args.test = get_train_test_datasets(args)
    
    
    args.config = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        push_to_hub=args.save_model_weights,
        hub_model_id=args.hub_model_id if args.save_model_weights else None,
        report_to="wandb",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
        per_device_eval_batch_size=1,
    )
    args.trainer = SFTTrainer(
        model=args.model,
        train_dataset=args.train,
        eval_dataset=args.test,
        args=args.config,
    )
    
    return args

def train(args: Args):
    args.trainer.train()
    if args.save_model_weights:
        args.trainer.push_to_hub()
    if args.save_local:
        local_path = Path("data/artifacts") / wandb.run.name
        local_path.mkdir(parents=True, exist_ok=True)
        args.trainer.save_model(local_path)
        args.tokenizer.save_pretrained(local_path)
        print(f"Model saved locally to {local_path}")
    wandb.finish()
    

def main():
    args = Args()
    args = init(args)
    train(args)
    
if __name__ == "__main__":
    main()
