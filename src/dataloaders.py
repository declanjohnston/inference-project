from datasets import Dataset, load_dataset
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import Args

def format_data(args: "Args", example: dict) -> dict:
    """
    Format the data into a prompt and completion format compatible with TRL SFTTrainer.

    Splits the pre-formatted `text` field at the response marker (e.g., "### Response:\n"),
    creating separate `prompt` and `completion` columns. This format enables completion-only
    loss computation during training, so the model learns to generate responses rather than
    memorizing the instruction template.

    Args:
        args: Configuration object containing `split_marker` (e.g., "### Response:\n").
        example: A single dataset example with a `text` field containing the full
            instruction-response sequence.

    Returns:
        Dict with `prompt` (instruction + marker) and `completion` (response) keys.
    """
    text = example["text"]
    parts = text.split(args.split_marker, 1)
    return {"prompt": parts[0] + args.split_marker, "completion": parts[1]}

def get_train_test_datasets(args: "Args") -> tuple[Dataset, Dataset]:
    dataset = load_dataset(args.dataset_name)
    dataset = dataset.map(lambda ex: format_data(args, ex), remove_columns=dataset["train"].column_names)
    split_dataset = dataset["train"].train_test_split(test_size=0.1)
    return split_dataset["train"], split_dataset["test"]

def get_validation_dataset(args: "Args") -> Dataset:
    return None