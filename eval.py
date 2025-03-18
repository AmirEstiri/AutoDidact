from unsloth import FastLanguageModel
import rl_helpers
from vllm import SamplingParams

# Load the model
model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",
    model_name = "checkpoints/checkpoint-10",
    max_seq_length = 64000,  # Match the training configuration
    load_in_4bit = True,     # Use 4-bit quantization for efficiency
    fast_inference = True,   # Enable vLLM fast inference
)


verifier_sampling_params = SamplingParams(
    temperature = 0.1,
    top_p = 0.95,
    max_tokens = 32000,
)


sampling_params = SamplingParams(
    temperature = 0.5,
    top_p = 0.95,
    max_tokens = 32000,
)

def verifier_generate_fn(inputs):
    return model.fast_generate(
        inputs,
        sampling_params = verifier_sampling_params,
    )
reward_correctness = rl_helpers.build_reward_correctness_fn(verifier_generate_fn, tokenizer)

def eval_generate_fn(inputs):
    return model.fast_generate(
        inputs,
        sampling_params=sampling_params,
    )


rl_helpers.run_eval(
    generate_fn=eval_generate_fn,
    verify_fn=reward_correctness,
    tokenizer=tokenizer,
)