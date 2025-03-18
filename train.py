from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel
from vllm import SamplingParams

import UnslothGRPOTrainerTemp
from rl_helpers import get_qa_dataset, run_agent, build_reward_correctness_fn, reward_formatting

train_dataset, test_dataset = get_qa_dataset()

max_seq_length = 64000 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.8, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth", # Enable long context finetuning
    random_state=7807,
)

training_args = UnslothGRPOTrainerTemp.UnslothGRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    use_agentic_generate = True, # use agentic generation
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 1024,
    max_completion_length = 4096,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 500,
    save_steps = 100,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "checkpoints",
)

model.agentic_generate = lambda prompts, generate_fn, max_generations=6: run_agent(
    generate_fn, tokenizer, prompts, max_generations
)

reward_correctness = build_reward_correctness_fn(
    lambda inputs: model.fast_generate(
        inputs, 
        sampling_params = SamplingParams(
            temperature = 0.1,
            top_p = 0.95,
            max_tokens = 32000,
        )
    ), 
    tokenizer
)

trainer = UnslothGRPOTrainerTemp.UnslothGRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        reward_correctness,
        reward_formatting,
    ],
    args = training_args,
    train_dataset = train_dataset,
)

trainer.train()