import argparse
import logging
import os
import torch

from imports import *
from config import MODEL_NAME
from data_processing import load_math_dataset, validate_dataset
from model_training import train_r1_zero, train_r1_sft, initialize_model, initialize_tokenizer
from inference import demo_r1_zero, demo_r1_sft, test_model_inference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if the environment is properly set up."""
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"Transformers version: {transformers.__version__}")
    logger.info(f"Using model: {MODEL_NAME}")

def test_base_model():
    """Test the base model to ensure it's working properly."""
    tokenizer = initialize_tokenizer()
    model, device = initialize_model()
    
    logger.info(f"Using device: {device}")
    logger.info(f"Model parameters: {model.num_parameters():,}")
    
    # Test basic inference
    test_input = "how are you?"
    response = test_model_inference(test_input, model, tokenizer, device)
    
    logger.info(f"Test Input: {test_input}")
    logger.info(f"Model Response: {response}")
    
    return model, tokenizer, device

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and test DeepSeek R1 models")
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="test_base",
        choices=["test_base", "train_r1_zero", "train_r1_sft", "demo_r1_zero", "demo_r1_sft", "all"],
        help="Mode to run the script in"
    )
    
    parser.add_argument(
        "--r1_zero_path",
        type=str,
        default="data/Qwen-GRPO-training",
        help="Path to save or load R1 Zero model"
    )
    
    parser.add_argument(
        "--r1_sft_path",
        type=str,
        default="data/Qwen-SFT-training",
        help="Path to save or load R1 SFT model"
    )
    
    return parser.parse_args()

def main():
    """Main function to run the script."""
    args = parse_args()
    
    # Check environment setup
    check_environment()
    
    if args.mode == "test_base" or args.mode == "all":
        logger.info("Testing base model...")
        test_base_model()
    
    if args.mode == "train_r1_zero" or args.mode == "all":
        logger.info("Training R1 Zero model...")
        train_r1_zero()
    
    if args.mode == "train_r1_sft" or args.mode == "all":
        logger.info("Training R1 SFT model...")
        train_r1_sft(args.r1_sft_path)
    
    if args.mode == "demo_r1_zero" or args.mode == "all":
        logger.info("Running R1 Zero demo...")
        demo_r1_zero(args.r1_zero_path)
    
    if args.mode == "demo_r1_sft" or args.mode == "all":
        logger.info("Running R1 SFT demo...")
        demo_r1_sft(args.r1_sft_path)

if __name__ == "__main__":
    main()