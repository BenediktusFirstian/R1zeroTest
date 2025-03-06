# DeepSeek-R1 Implementation Module
# This module provides a step-by-step implementation of the DeepSeek-R1 training methodology

from .imports import *
from .config import (
    MODEL_NAME, 
    OUTPUT_DIR, 
    SYSTEM_PROMPT, 
    GRPOScriptArguments, 
    ModelConfig,
    training_args, 
    get_sft_training_args
)
from .data_processing import (
    make_conversation,
    load_math_dataset,
    validate_dataset,
    load_bespoke_dataset,
    generate_few_shot_examples,
    generate_direct_prompt,
    refine_output
)
from .reward_functions import (
    accuracy_reward,
    format_reward,
    reasoning_steps_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    get_language_consistency_reward,
    get_reward_functions
)
from .model_training import (
    initialize_tokenizer,
    initialize_model,
    train_r1_zero,
    train_r1_sft
)
from .inference import (
    test_model_inference,
    test_trained_model_inference,
    demo_r1_zero,
    demo_r1_sft
)

__version__ = "0.1.0"
__author__ = "DeepSeek Implementation Team"