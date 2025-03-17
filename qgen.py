import re
import json
from typing import List, Tuple, Optional, Dict

# Setup Llama backend via unsloth and vLLM
from unsloth import FastLanguageModel
from vllm import SamplingParams


def batch_generate(prompts: List[str]) -> List[str]:
    """
    Given a list of prompt strings, returns a list of generated outputs.
    """

    def format_input(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )

    formatted = [format_input(p) for p in prompts]
    outputs = model.fast_generate(formatted, sampling_params=sampling_params)
    return [output.outputs[0].text for output in outputs]


def parse_qa_block(block: str) -> Optional[Tuple[str, str, str]]:
    """
    Parses a QA block that should contain exactly three non-empty lines:
      - A line starting with "Question:"
      - A line starting with "Answer:"
      - A line starting with "Difficulty:"

    If the markers are not present but the block contains exactly three lines,
    those are used in order.

    Returns a tuple (question, answer, difficulty) or None if parsing fails.
    """
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return None

    question, answer, difficulty = None, None, None
    for line in lines:
        lower = line.lower()
        if question is None and lower.startswith("question:"):
            question = line[len("question:") :].strip()
        elif answer is None and lower.startswith("answer:"):
            answer = line[len("answer:") :].strip()
        elif difficulty is None and lower.startswith("difficulty:"):
            difficulty = line[len("difficulty:") :].strip()

    if question and answer and difficulty:
        return question, answer, difficulty
    if len(lines) == 3:
        return lines[0], lines[1], lines[2]
    return None


def parse_multiple_qa_output(output: str) -> List[Tuple[str, str, str]]:
    """
    Splits the output into blocks (separated by one or more blank lines) and
    attempts to parse each as a QA pair.

    Returns a list of successfully parsed QA tuples.
    """
    blocks = re.split(r"\n\s*\n", output.strip())
    qa_pairs = []
    for block in blocks:
        parsed = parse_qa_block(block)
        if parsed:
            qa_pairs.append(parsed)
    return qa_pairs


def generate_question_batch_for_chunks(
    chunks: List, num_questions: int = 2, context_size: int = 2
) -> List[Dict]:
    """
    Generates QA pairs for multiple chunks in batch.

    For each chunk (except the first and last), a sliding window is used for context:
      - before: previous chunk's content
      - current: current chunk's content
      - after: next chunk's content

    Each prompt instructs the model to output exactly three lines per QA pair with markers.
    Failed prompts are retried once in batch; if still unsuccessful, they are skipped.

    Returns a list of dicts with keys: "chunk_id", "question", "answer", "difficulty".
    """
    prompts = []
    chunk_ids = []
    data_chunk_ids = list(chunks.keys())
    data_text_chunks = list(chunks.values())

    # Prepare prompts using a sliding window
    for i in range(1, len(data_text_chunks) - 1):
        before = data_text_chunks[i - context_size:i]
        current = data_text_chunks[i]
        after = data_text_chunks[i + 1:i + context_size]
        prompt = (
            f"You are a hardware and electrical engineering expert. You are given a technical text and asked to generate questions and answers from it.\n"
            f"From the tecnical text within ==BEGIN== and ==END==, generate {num_questions} questions with answers.\n"
            "For each QA pair, output exactly two lines with no extra commentary:\n"
            "Line 1: Question: <your question>\n"
            "Line 2: Answer: <the answer>\n"
            "Do not include any additional text.\n\n"
            "IMPORTANT: All of the questions must be different from each other.\n"
            "==BEGIN==\n"
            f"{before}\n{current}\n{after}\n"
            "==END==\n"
        )
        prompts.append(prompt)
        chunk_ids.append(data_chunk_ids[i])

    # First batch generation
    outputs = batch_generate(prompts)
    results = [None] * len(outputs)
    failed_indices = []

    # Parse each output
    for idx, output in enumerate(outputs):
        qa_pairs = parse_multiple_qa_output(output)
        if qa_pairs is None or len(qa_pairs) < num_questions:
            failed_indices.append(idx)
        else:
            results[idx] = qa_pairs[:num_questions]

    final_questions = []
    for i, qa_list in enumerate(results):
        if qa_list is not None:
            for qa in qa_list:
                final_questions.append(
                    {
                        "chunk_id": chunk_ids[i],
                        "question": qa[0],
                        "answer": qa[1],
                    }
                )
    json.dump(final_questions, open("data/questions.json", "w"), indent=4)

    # Retry failed prompts in batch
    if failed_indices:
        print(f"Retrying {len(failed_indices)} failed prompt(s)...")
        retry_prompts = [prompts[i] for i in failed_indices]
        retry_outputs = batch_generate(retry_prompts)
        for j, idx in enumerate(failed_indices):
            qa_pairs = parse_multiple_qa_output(retry_outputs[j])
            if qa_pairs is not None and len(qa_pairs) >= num_questions:
                results[idx] = qa_pairs[:num_questions]
            else:
                results[idx] = None  # Mark as failed

    # Build final output, skipping prompts that failed even after retry
    final_questions = []
    for i, qa_list in enumerate(results):
        if qa_list is not None:
            for qa in qa_list:
                final_questions.append(
                    {
                        "chunk_id": chunk_ids[i],
                        "question": qa[0],
                        "answer": qa[1],
                    }
                )
    json.dump(final_questions, open("data/questions.json", "w"), indent=4)



if __name__ == "__main__":
    chunks = json.load(open("data/chunks.json", "rb"))
    
    # Load the Llama model (adjust parameters as needed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
        max_seq_length=32000,
        load_in_4bit=True,  # Use 4-bit quantization if desired
        fast_inference=True,  # Enable fast inference
        gpu_memory_utilization=0.8,  # Adjust based on your GPU memory
    )

    # Define sampling parameters for generation
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.95,
        max_tokens=64000,
    )

    # Generate QA pairs in batch (using a sliding window over the chunks)
    generate_question_batch_for_chunks(
        chunks, num_questions=8, context_size=5
    )
