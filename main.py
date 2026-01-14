from mlh.hypers import Hypers, TBD
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.model import get_base_model, get_tokenizer
from dataclasses import dataclass
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, load_dataset
from src.dataloaders import get_train_test_datasets, get_validation_dataset
@dataclass
class Args(Hypers):
    
    # Model and dataset configuration
    model_name: str = "google/gemma-2-2b"
    dataset_name: str = "tatsu-lab/alpaca"
    split_marker: str = "### Response:\\n"
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
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 1024
    max_length: int = 1024
    output_dir: str = "./sft_output"
    max_steps: int = 10
    logging_steps: int = 10
    save_steps: int = 5
    eval_steps: int = 5
    
    
def init(args: Args):
    args.model = get_base_model(args)
    args.tokenizer = get_tokenizer(args)
    args.train, args.test = get_train_test_datasets(args)
    args.validation = get_validation_dataset(args)
    
    
    args.config = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
    )
    args.trainer = SFTTrainer(
        model=args.model,
        train_dataset=args.train,
        eval_dataset=args.validation,
        config=args.config,
    )
    
    return args

def train(args: Args):
    args.trainer.train()
    

def main():
    args = Args()
    args = init(args)
    train(args)
    
if __name__ == "__main__":
    main()
