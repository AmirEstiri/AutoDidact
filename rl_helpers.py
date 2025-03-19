"""
RL helpers module for handling tool-based conversations.
This module provides utility functions for handling chat-based tool interactions
and calculating rewards based on the quality of responses.
"""

import re
import torch
from search_module import search, get_qa_dataset
from dataclasses import dataclass
from typing import List, Callable


from trl.trainer.grpo_trainer import apply_chat_template


def get_initial_chat(question):
    return {
        "messages": [
            {
                "role": "system",
                "content": f"""When you receive a tool call response, use the output to format an answer to the original user question.
                You are an expert in the field of hardware and electrical engineering.
                Your answer should address the user's question directly. Do not write the tool call response in your answer.
                But you can give a concise summary of the tool call response that is relevant to the user's question only if it is necessary.
                """,
            },
            {
                "role": "user",
                "content": f"""You are a research assistant and expert in the field of hardware and electrical engineering, and you use the search_corpus tool to find answers to questions.
                Given a question, answer it using by doing searches using the search_corpus tool.
                To use the search_corpus tool, respond with a JSON for a function call with its proper arguments.

                You may also reason in any message, thinking step by step about how to answer the question. Wrap your reasoning in <reasoning> and </reasoning> tags.

                Question: {question}
                """,
            },
        ]
    }


def run_agent_generations(generate_fn, tokenizer, chat_states):
    prompts = []
    batch_indices = []
    for idx, chat_state in enumerate(chat_states):
        if chat_state.get("finished"):
            continue

        if chat_state["messages"][-1]["role"] == "user":
            prompt = apply_chat_template(chat_state, tokenizer=tokenizer)["text"]
            prompts.append(prompt)
            batch_indices.append(idx)

    if prompts:
        responses = generate_fn(prompts)
        breakpoint()
        for i, idx in enumerate(batch_indices):
            chat_state = chat_states[idx]
            full_response = responses[i].outputs[0].text
            assistant_response = full_response.split(
                "<|start_header_id|>assistant<|end_header_id|>"
            )[-1]
            chat_state["messages"].append(
                {"role": "assistant", "content": assistant_response}
            )
    return chat_states


def check_finished_chats(chat_states):
    for chat_state in chat_states:
        if chat_state.get("finished"):
            continue
        assert (
            chat_state["messages"][-1]["role"] == "assistant"
        ), "Expected the last role to be assistant"
        search_query = chat_state["messages"][-1]["content"]
        if search_query == "":
            chat_state["finished"] = True
    return chat_states


def run_tool_calls(chat_states):
    """
    Execute tool calls found in chat states.

    Args:
        chat_states: List of chat states

    Returns:
        list: Updated chat states with tool call results
    """
    for chat_state in chat_states:
        if chat_state.get("finished"):
            continue
        assert (
            chat_state["messages"][-1]["role"] == "assistant"
        ), "Expected the last role to be assistant to run tool calls"
        try:
            search_query = chat_state["messages"][-1]["content"]
            if search_query == "":
                raise ValueError("Expected a search query in assistant response")
            results = search(search_query, return_type=str, results=10)
            chat_state["messages"].append({"role": "user", "content": results})
        except Exception as e:
            chat_state["messages"].append(
                {"role": "system", "content": f"Error during post-processing: {str(e)}"}
            )
            chat_state["finished"] = True
    return chat_states


def get_mask(text, tokenizer):
    encoding = tokenizer(text, add_special_tokens=False)
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    assistant_token = tokenizer.convert_tokens_to_ids("assistant")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    assistant_ranges = []
    i = 0
    while i < len(encoding.input_ids) - 1:
        if (
            encoding.input_ids[i] == start_header_id
            and encoding.input_ids[i + 1] == assistant_token
        ):
            i += 2
            while (
                i < len(encoding.input_ids) and encoding.input_ids[i] != end_header_id
            ):
                i += 1
            i += 2
            start_idx = i
            while i < len(encoding.input_ids) and encoding.input_ids[i] != eot_id:
                i += 1
            end_idx = i
            assistant_ranges.append((start_idx, end_idx))
        else:
            i += 1
    mask = [0] * len(encoding.input_ids)
    for start_idx, end_idx in assistant_ranges:
        for idx in range(start_idx, end_idx):
            mask[idx] = 1
    return torch.tensor(mask, dtype=torch.int)


def check_exceeded_max_new_tokens(chat_states, max_new_tokens, tokenizer):
    for chat_state in chat_states:
        if chat_state.get("finished"):
            continue
        initial_length = chat_state["initial_length"]
        new_length = get_chat_num_tokens(chat_state, tokenizer)
        if new_length - initial_length > max_new_tokens:
            chat_state["finished"] = True
    return chat_states


@dataclass
class AgenticOutputs:
    prompt_tokens: list[torch.Tensor]
    response_tokens: list[torch.Tensor]
    response_masks: list[torch.Tensor]
    final_response_str: list[str]
    full_chat_states: list[dict]


def get_chat_num_tokens(chat_state, tokenizer):
    chat_text = apply_chat_template(chat_state, tokenizer=tokenizer)["text"]
    return (
        tokenizer(chat_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        .squeeze()
        .shape[0]
    )


def run_agent(
    generate_fn, tokenizer, questions, max_generations=5, max_new_tokens=4096
):
    """
    Run the agent to completion for a batch of questions.

    Args:
        generate_fn: Function to generate model responses
        tokenizer: Tokenizer for processing text
        batch: Batch of data containing questions
        max_generations: Maximum number of generation steps

    Returns:
        list: Final answers for each question
    """
    chat_states = [get_initial_chat(question) for question in questions]
    # set the initial_prompt length
    for chat_state in chat_states:
        chat_state["initial_length"] = get_chat_num_tokens(chat_state, tokenizer)

    # agent loop
    for i in range(max_generations):
        chat_states = run_agent_generations(generate_fn, tokenizer, chat_states)
        chat_states = check_finished_chats(chat_states)
        chat_states = run_tool_calls(chat_states)
        chat_states = check_exceeded_max_new_tokens(
            chat_states, max_new_tokens, tokenizer
        )

    answers = []
    for chat in chat_states:
        answers.append(chat["messages"][-1]["content"])

    def split_prompt_assistant(convo_text):
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        idx = convo_text.find(marker)
        if idx == -1:
            raise ValueError("Could not find assistant marker in conversation text.")
            return convo_text, ""
        # Include the marker in the prompt by slicing up to the end of the marker.
        prompt = convo_text[: idx + len(marker)]
        # The assistant response is everything after the marker.
        assistant_response = convo_text[idx + len(marker) :]
        return prompt, assistant_response

    str_chats = [
        apply_chat_template(chat, tokenizer=tokenizer)["text"] for chat in chat_states
    ]
    prompt_toks, response_toks, response_masks = [], [], []
    for str_chat in str_chats:
        prompt, response = split_prompt_assistant(str_chat)
        prompt_toks.append(
            tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ].squeeze()
        )
        response_toks.append(
            tokenizer(response, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ].squeeze()[:max_new_tokens]
        )
        mask = get_mask(str_chat, tokenizer)[len(prompt_toks[-1]) :][:max_new_tokens]

        response_masks.append(mask)

    final_response_str = [chat["messages"][-1]["content"] for chat in chat_states]
    full_chat_states = chat_states
    agentic_outputs = AgenticOutputs(
        prompt_tokens=prompt_toks,
        response_tokens=response_toks,
        response_masks=response_masks,
        final_response_str=final_response_str,
        full_chat_states=full_chat_states,
    )

    return agentic_outputs


def check_student_answers(
    questions: List[str],
    answers: List[str],
    student_answers: List[str],
    vllm_generate_func: Callable[[List[str]], List[str]],
    tokenizer,
    log_file: str = "qa_log.txt",
) -> List[bool]:
    """
    Evaluates a list of student answers against the true answers using a vLLM generate function.
    The function applies the chat template to each prompt before passing it to the generate function.
    It also appends the details of each QA pair and the verifier's response to a log file.

    Args:
        questions: A list of strings representing the questions.
        answers: A list of strings representing the correct answers.
        student_answers: A list of strings containing the student's answers.
        vllm_generate_func: A function that takes a list of chat-formatted prompt strings and returns a list of generated outputs.
        tokenizer: The tokenizer used to apply the chat template.
        log_file: Optional; path to the file where the QA pairs and verification responses will be appended.

    Returns:
        A list of booleans indicating whether each student's answer is correct.
    """
    if not (len(questions) == len(answers) == len(student_answers)):
        raise ValueError(
            "The number of questions, answers, and student answers must be equal."
        )

    prompts = []
    for question, answer, student_ans in zip(questions, answers, student_answers):
        # Construct the plain text prompt for each QA pair.
        prompt_text = (
            "You are grading a student's answer. For the following question, "
            "compare the student's answer to the correct answer. Reply with 'Yes' if the student's answer is correct, or 'No' if it is completely incorrect.\n\n"
            f"Question: {question}\n"
            f"Correct Answer: {answer}\n"
            f"Provided Answer: {student_ans}\n"
        )
        # Apply the chat template to the prompt.
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(formatted_prompt)

    # Get the model responses in batch (each response should ideally be "Yes" or "No")
    responses = vllm_generate_func(prompts)
    responses_text = [response.outputs[0].text for response in responses]

    # Evaluate each response and mark as correct if "yes" appears in the answer (case-insensitive)
    results = []
    for response in responses_text:
        results.append("yes" in response.lower())

    # Append the QA details and verifier's response to the specified log file
    with open(log_file, "a") as file:
        for question, answer, student_ans, verifier_response in zip(
            questions, answers, student_answers, responses_text
        ):
            file.write("Question: " + question + "\n")
            file.write("Correct Answer: " + answer + "\n")
            file.write("Student Answer: " + student_ans + "\n")
            file.write("Verifier said: " + verifier_response + "\n")
            file.write("-" * 40 + "\n")

    return results


def build_reward_correctness_fn(generate_fn, tokenizer):
    def reward_correctness(prompts, completions, **reward_kwargs):
        teacher_answers = reward_kwargs["answer"]
        student_answers = [
            completion["messages"][-1]["content"] for completion in completions
        ]

        correct = check_student_answers(
            prompts,
            teacher_answers,
            student_answers,
            vllm_generate_func=generate_fn,
            tokenizer=tokenizer,
        )
        return correct

    return reward_correctness


def reward_formatting(prompts, completions, **reward_kwargs):
    # make sure full chats doesn't have any error function calls
    has_error = [False] * len(completions)
    for i, chat in enumerate(completions):
        for message in chat["messages"]:
            if "Error during" in message["content"]:
                has_error[i] = True
                break
    return [0.7 if not e else 0 for e in has_error]


def run_eval(generate_fn, verify_fn, tokenizer):
    _, test_dataset = get_qa_dataset()
    questions = test_dataset["prompt"]
    agentic_outputs = run_agent(generate_fn, tokenizer, questions)
    full_chat_states = agentic_outputs.full_chat_states
    # final_responses = agentic_outputs.final_response_str
    rewards = verify_fn(questions, full_chat_states, answer=test_dataset["answer"])

    print("RESULTS:")
    print("percentage of correct answers:", sum(rewards) / len(rewards))
    print("=" * 30)
    return full_chat_states
