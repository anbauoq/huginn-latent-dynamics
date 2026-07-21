"""HuginnAdapter: the only module that touches Huginn-specific internals.

Ordinary `output_hidden_states=True` does not surface Huginn's recurrent
per-iteration states -- `iterate_forward` (in the remote `raven_modeling_minimal.py`)
only returns the *final* recurrent state. To capture every intermediate step
without forking the model, this module temporarily monkey-patches
`core_block_forward`, the one method invoked exactly once per recurrent
iteration, for the duration of a single teacher-forced forward pass.
"""

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_NAME_DEFAULT = "tomg-group-umd/huginn-0125"


@dataclass
class GenerationResult:
    prompt_ids: list[int]
    generated_ids: list[int]
    generated_text: str


def resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(device: str) -> torch.dtype:
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


@contextlib.contextmanager
def _capture_core_block_states(model: torch.nn.Module) -> Iterator[list[torch.Tensor]]:
    """Record the latent state before and after every recurrent core-block call.

    `core_block_forward(x, input_embeds, freqs_cis, mask, past_key_values,
    block_idx, current_step)` takes the incoming state `x` and returns the
    state after one application of the recurrent block. Wrapping it captures
    the initial state (current_step == 0, before the first application) plus
    the state after each of the `num_steps` applications -- `num_steps + 1`
    states in total, in order.
    """
    states: list[torch.Tensor] = []
    original = model.core_block_forward

    def capturing(x, input_embeds, freqs_cis, mask, past_key_values, block_idx, current_step):
        if current_step == 0:
            states.append(x.detach().to(torch.float32).cpu())
        out, new_block_idx = original(x, input_embeds, freqs_cis, mask, past_key_values, block_idx, current_step)
        states.append(out.detach().to(torch.float32).cpu())
        return out, new_block_idx

    model.core_block_forward = capturing
    try:
        yield states
    finally:
        del model.core_block_forward


class HuginnAdapter:
    """Loads Huginn, runs greedy generation, and captures recurrent trajectories.

    This is the only class in the package that knows about Huginn's chat
    template, its custom `generate` dispatch, or its recurrent core-block loop.
    """

    def __init__(self, model_name: str = MODEL_NAME_DEFAULT, device: Optional[str] = None):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.dtype, trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()

    @property
    def hidden_size(self) -> int:
        return self.model.config.n_embd

    def build_prompt_ids(self, question: str, system_prompt: str) -> list[int]:
        """Render the chat template (system + user turn) and tokenize it."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer.encode(chat_text, add_special_tokens=False)

    def generate(self, prompt_ids: list[int], num_steps: int, max_new_tokens: int, seed: int) -> GenerationResult:
        """Deterministic greedy generation. `num_steps` is passed directly to the model."""
        torch.manual_seed(seed)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        generation_config = GenerationConfig.from_pretrained(self.model_name)
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_k = None
        generation_config.top_p = None
        generation_config.return_dict_in_generate = True

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                generation_config,
                num_steps=num_steps,
                max_new_tokens=max_new_tokens,
                tokenizer=self.tokenizer,
                stop_strings=["<|end_text|>", "<|end_turn|>"],
            )

        sequence = output.sequences[0].tolist()
        generated_ids = sequence[len(prompt_ids) :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return GenerationResult(prompt_ids=list(prompt_ids), generated_ids=generated_ids, generated_text=generated_text)

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def special_token_ids(self) -> set[int]:
        return set(self.tokenizer.all_special_ids)

    def capture_trajectory(self, full_ids: list[int], num_steps: int) -> np.ndarray:
        """Teacher-forced forward pass over `full_ids`, capturing every recurrent state.

        Returns an array of shape [num_steps + 1, sequence_length, hidden_size].
        Index 0 is the initial recurrent state (available before any core-block
        application); index i (i >= 1) is the state after the i-th application
        of the recurrent core block.
        """
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        output_details = {
            "return_logits": False,
            "return_latents": False,
            "return_head": False,
            "return_stats": False,
        }
        with torch.no_grad(), _capture_core_block_states(self.model) as captured:
            self.model(input_ids=input_ids, num_steps=num_steps, use_cache=False, output_details=output_details)

        stacked = torch.stack(captured, dim=0)  # [num_steps + 1, 1, seq_len, hidden]
        return stacked.squeeze(1).numpy()
