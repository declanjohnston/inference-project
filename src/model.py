# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import Args



def get_base_model(args: "Args"):
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    return model

def get_tokenizer(args: "Args"):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    return tokenizer
