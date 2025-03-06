from imports import *
from config import MODEL_NAME, OUTPUT_DIR, training_args, get_sft_training_args, GRPOScriptArguments, ModelConfig
from data_processing import load_math_dataset, load_bespoke_dataset, validate_dataset
from reward_functions import get_reward_functions

logger = logging.getLogger(__name__)

class LoggingCallback(TrainerCallback):
    """
    A simple callback for logging training information at specific steps.
    """
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % args.logging_steps == 0:
            logger.info(f"Step {state.global_step}: Loss = {state.log_history[-1].get('loss', None)}, Learning Rate = {state.log_history[-1].get('learning_rate', None)}")

def get_callbacks(training_args, model_args, script_args):
    """
    Returns a list of callbacks to be used during training.
    For now, it includes only the LoggingCallback. You can extend this to add more callbacks.
    """
    callbacks = [LoggingCallback()] # Instantiate our LoggingCallback
    return callbacks

def initialize_tokenizer(model_name=MODEL_NAME):
    """Initialize tokenizer with appropriate settings."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer

def initialize_model(model_name=MODEL_NAME, device=None):
    """Initialize model with appropriate settings."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # Move model to device
    model.to(device)
    
    return model, device

def train_r1_zero():
    """Train DeepSeek R1 Zero using GRPO algorithm directly on the base model."""
    # Initialize configurations
    script_args = GRPOScriptArguments()
    model_args = ModelConfig()
    
    # Load dataset
    dataset = load_math_dataset()
    validate_dataset(dataset)
    
    # Initialize tokenizer and model
    tokenizer = initialize_tokenizer()
    model, device = initialize_model()
    
    # Get reward functions and callbacks
    reward_functions = get_reward_functions(script_args)
    callbacks = get_callbacks(training_args, model_args, script_args)
    
    # Create GRPOConfig from TrainingArguments
    grpo_config = GRPOConfig(**training_args.to_dict())
    
    # Initialize GRPO Trainer
    grpo_trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=grpo_config,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        callbacks=callbacks
    )
    
    # Start training
    train_result = grpo_trainer.train()
    
    # Save trained model
    tokenizer.save_pretrained(OUTPUT_DIR)
    grpo_trainer.save_model(OUTPUT_DIR)
    
    print(f"GRPO Trained model saved to {OUTPUT_DIR}")
    
    return grpo_trainer, tokenizer

def train_r1_sft(sft_output_dir="data/Qwen-SFT-training"):
    """Train DeepSeek R1 using SFT on a cold start dataset."""
    # Create output directory if it doesn't exist
    os.makedirs(sft_output_dir, exist_ok=True)
    
    # Load bespoke dataset for SFT
    dataset_sft = load_dataset("bespokelabs/Bespoke-Stratos-17k", split='train')
    
    # Initialize tokenizer and model
    tokenizer = initialize_tokenizer()
    model_sft, _ = initialize_model()
    
    # Get SFT training arguments
    sft_args = get_sft_training_args(sft_output_dir)
    
    # Initialize SFT Trainer
    sft_trainer = SFTTrainer(
        model=model_sft,
        train_dataset=dataset_sft,
        tokenizer=tokenizer,
        args=sft_args,
    )
    
    # Start SFT training
    sft_train_result = sft_trainer.train()
    
    # Save trained model
    tokenizer.save_pretrained(sft_output_dir)
    sft_trainer.save_model(sft_output_dir)
    
    print(f"SFT Trained model saved to {sft_output_dir}")
    
    return sft_trainer, tokenizer