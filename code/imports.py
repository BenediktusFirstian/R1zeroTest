# Import necessary libraries
import logging
import os
import sys
import re
import math
from dataclasses import dataclass, field
from typing import List, Optional

# Import PyTorch and Hugging Face Transformers
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import get_last_checkpoint

# Import dataset utilities
import datasets
from datasets import load_dataset

# Import libraries from TRL (Transformers Reinforcement Learning)
from trl import (
    AutoModelForCausalLMWithValueHead, 
    PPOConfig, 
    PPOTrainer, 
    GRPOTrainer, 
    GRPOConfig, 
    SFTTrainer
)

# Import math-related utilities
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
