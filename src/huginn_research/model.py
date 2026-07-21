"""HuginnAdapter: the only module that touches Huginn-specific internals.

Ordinary `output_hidden_states=True` does not surface Huginn's recurrent
per-iteration states -- `iterate_forward` (in the remote `raven_modeling_minimal.py`)
only returns the *final* recurrent state. To capture every intermediate step
without forking the model, this module temporarily monkey-patches
`core_block_forward`, the one method invoked exactly once per recurrent
iteration.

Two capture modes build on that same hook:

* `capture_trajectory` -- a single teacher-forced forward pass over a fixed
  sequence, recording every position's state at every recurrent step.
* `generate_with_trajectory` -- captures states while `model.generate()` runs.
  Each of `generate()`'s internal forward calls processes exactly one causal
  position that matters (the whole prompt on the prefill call, one new token
  on every decoding step after that), and predicts exactly one output token.
  Recording only that position's states, grouped by forward call, yields one
  prediction-aligned trajectory per generated token with no second pass.
"""

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_NAME_DEFAULT = "tomg-group-umd/huginn-0125"

StepObserver = Callable[[torch.Tensor, torch.Tensor, int], None]


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
def _patched_core_block_forward(model: torch.nn.Module, on_step: StepObserver) -> Iterator[None]:
    """Temporarily wrap `core_block_forward` so `on_step(x_in, x_out, current_step)`
    fires for every recurrent-step application, then restore the original method.

    `core_block_forward(x, input_embeds, freqs_cis, mask, past_key_values,
    block_idx, current_step)` takes the incoming state `x_in` and returns the
    state `x_out` after one application of the recurrent block. `current_step`
    resets to 0 at the start of every `model.forward()` call -- the only
    reliable signal that a new group of recurrent states has begun.
    """
    original = model.core_block_forward

    def wrapped(x, input_embeds, freqs_cis, mask, past_key_values, block_idx, current_step):
        out, new_block_idx = original(x, input_embeds, freqs_cis, mask, past_key_values, block_idx, current_step)
        on_step(x, out, current_step)
        return out, new_block_idx

    model.core_block_forward = wrapped
    try:
        yield
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

    def _build_generation_config(self) -> GenerationConfig:
        generation_config = GenerationConfig.from_pretrained(self.model_name)
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_k = None
        generation_config.top_p = None
        generation_config.return_dict_in_generate = True
        return generation_config

    def _generate_raw(self, prompt_ids: list[int], num_steps: int, max_new_tokens: int, seed: int):
        """Run `model.generate()` with deterministic greedy decoding. `num_steps`
        is passed directly to the model, not through the GenerationConfig."""
        torch.manual_seed(seed)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generation_config = self._build_generation_config()
        with torch.no_grad():
            return self.model.generate(
                input_ids,
                generation_config,
                num_steps=num_steps,
                max_new_tokens=max_new_tokens,
                tokenizer=self.tokenizer,
                stop_strings=["<|end_text|>", "<|end_turn|>"],
            )

    def _to_generation_result(self, prompt_ids: list[int], output) -> GenerationResult:
        sequence = output.sequences[0].tolist()
        generated_ids = sequence[len(prompt_ids) :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return GenerationResult(prompt_ids=list(prompt_ids), generated_ids=generated_ids, generated_text=generated_text)

    def generate(self, prompt_ids: list[int], num_steps: int, max_new_tokens: int, seed: int) -> GenerationResult:
        """Deterministic greedy generation, without capturing recurrent states."""
        output = self._generate_raw(prompt_ids, num_steps, max_new_tokens, seed)
        return self._to_generation_result(prompt_ids, output)

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def special_token_ids(self) -> set[int]:
        return set(self.tokenizer.all_special_ids)

    def capture_trajectory(self, full_ids: list[int], num_steps: int) -> np.ndarray:
        """Teacher-forced forward pass over `full_ids`, capturing every position's
        recurrent state at every step.

        Returns an array of shape [num_steps + 1, sequence_length, hidden_size].
        Index 0 is the initial recurrent state (available before any core-block
        application); index i (i >= 1) is the state after the i-th application
        of the recurrent core block.
        """
        states: list[torch.Tensor] = []

        def on_step(x_in: torch.Tensor, x_out: torch.Tensor, current_step: int) -> None:
            if current_step == 0:
                states.append(x_in.detach().to(torch.float32).cpu())
            states.append(x_out.detach().to(torch.float32).cpu())

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        output_details = {
            "return_logits": False,
            "return_latents": False,
            "return_head": False,
            "return_stats": False,
        }
        with torch.no_grad(), _patched_core_block_forward(self.model, on_step):
            self.model(input_ids=input_ids, num_steps=num_steps, use_cache=False, output_details=output_details)

        stacked = torch.stack(states, dim=0)  # [num_steps + 1, 1, seq_len, hidden]
        return stacked.squeeze(1).numpy()

    def generate_with_trajectory(
        self, prompt_ids: list[int], num_steps: int, max_new_tokens: int, seed: int
    ) -> tuple[GenerationResult, np.ndarray]:
        """Greedy generation that captures recurrent states as `generate()` runs,
        with no second forward pass.

        Every forward call `generate()` makes processes exactly one causal
        position whose logits are used (the last prompt position on the
        prefill call, the single new token on every decoding step after that)
        and predicts exactly one generated token, so forward calls and
        generated tokens are in 1:1 correspondence. Grouping captured states
        by forward call and keeping only the last active sequence position of
        each group yields one prediction-aligned trajectory per generated
        token.

        Returns `(result, states)` where `states` has shape
        [num_steps + 1, len(result.generated_ids), hidden_size].
        """
        groups: list[list[torch.Tensor]] = []

        def on_step(x_in: torch.Tensor, x_out: torch.Tensor, current_step: int) -> None:
            if current_step == 0:
                groups.append([x_in[:, -1, :].detach().to(torch.float32).cpu()])
            groups[-1].append(x_out[:, -1, :].detach().to(torch.float32).cpu())

        with _patched_core_block_forward(self.model, on_step):
            output = self._generate_raw(prompt_ids, num_steps, max_new_tokens, seed)

        result = self._to_generation_result(prompt_ids, output)
        if len(groups) != len(result.generated_ids):
            raise RuntimeError(
                f"captured {len(groups)} recurrent forward-call groups but generate() produced "
                f"{len(result.generated_ids)} tokens; generation-mode trajectory capture requires "
                "exactly one captured group per generated token and will not guess a mapping between them."
            )

        per_token_states = [torch.stack(group, dim=0).squeeze(1) for group in groups]  # each [num_steps + 1, hidden]
        states = torch.stack(per_token_states, dim=1).numpy()  # [num_steps + 1, num_generated, hidden]
        return result, states
