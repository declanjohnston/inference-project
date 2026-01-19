from mlh.hypers import Hypers, TBD
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.model import get_base_model, get_tokenizer
from dataclasses import dataclass, asdict
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, load_dataset
from src.dataloaders import get_train_test_datasets, get_validation_dataset
import wandb
@dataclass
class Args(Hypers):
    
    # Project configuration
    project_name: str = "inference-project"
    hub_model_id: str = "djohnston5/gemma-2-2b-sft"
    save_model_weights: bool = False
    
    # Model and dataset configuration
    model_name: str = "google/gemma-2-2b"
    dataset_name: str = "tatsu-lab/alpaca"
    split_marker: str = "### Response:\n"
    # Placeholders for training objects
    model: AutoModelForCausalLM = TBD()
    tokenizer: AutoTokenizer = TBD()
    train: Dataset = TBD()
    test: Dataset = TBD()
    validation: Dataset = TBD()
    trainer: SFTTrainer = TBD()
    config: SFTConfig = TBD()
    
    # Hyperparameters
    learning_rate: float = 1e-4
    num_epochs: int = 5
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 1024
    max_length: int = 1024
    output_dir: str = "./sft_output"
    max_steps: int = 10000
    logging_steps: int = 10
    eval_steps: int = 50
    
    
def init(args: Args):
    wandb.init(
        entity="declanjohnston5-declan-johnston",
        project=args.project_name,
        config=args.to_dict(),
    )
    
    args.model = get_base_model(args)
    args.tokenizer = get_tokenizer(args)
    args.train, args.test, args.validation = get_train_test_datasets(args)
    
    
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
    )
    args.trainer = SFTTrainer(
        model=args.model,
        train_dataset=args.train,
        eval_dataset=args.validation,
        args=args.config,
    )
    
    return args

def train(args: Args):
    args.trainer.train()
    if args.save_model_weights:
        args.trainer.push_to_hub()
    wandb.finish()
    

def main():
    args = Args()
    args = init(args)
    train(args)
    
if __name__ == "__main__":
    main()
