"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Union

import numpy as np
from transformers import AutoProcessor, PreTrainedTokenizerBase

TOK_ACTION_START = "<|action_start|>"
TOK_ACTION_END = "<|action_end|>"


ENCODE = lambda x: f"<|a_{x}|>"


class FastActionTokenizer: 
    
    # input: action_data = np.random.rand(256, 50, 14)    # one batch of action chunks    
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        time_horizon : int,
        action_dim : int,
        bins: int = 2048,
        pretrained_checkpoint: str = "physical-intelligence/fast",
    ) -> None:
        # Load the tokenizer from the Hugging Face hub
        self.fast_tokenizer = AutoProcessor.from_pretrained(pretrained_checkpoint, trust_remote_code=True)
        self.fast_tokenizer.action_dim = action_dim
        self.fast_tokenizer.time_horizon = time_horizon
        self.n_bins = bins
        assert self.fast_tokenizer.vocab_size  == bins

        # Expand the tokenizer's vocabulary with new action tokens

        new_tokens = []
        for i in range(self.n_bins):
            new_tokens.append(ENCODE(i))
        tokenizer.add_tokens(new_tokens)

        special_tokens_dict = {
            "additional_special_tokens": [TOK_ACTION_START, TOK_ACTION_END]
        }
        tokenizer.add_special_tokens(special_tokens_dict)  # type: ignore

        self.action_token_begin_idx = tokenizer(new_tokens[0])["input_ids"][0]

    def __call__(self, action: np.ndarray, wrap_special_tokens: bool = False) -> Union[str, List[str]]:
        action_tokens = self.fast_tokenizer(action) # return list[list[int]] len=batch_size
 
        WRAP = lambda x: f"{TOK_ACTION_START}{x}{TOK_ACTION_END}" if wrap_special_tokens else x
        # print("Action tokens length:", len(action_tokens[0]))
        # Handle single element vs. batch
        if len(action_tokens) == 1:
            assert action.shape[0] == self.fast_tokenizer.time_horizon and \
                    action.shape[1] == self.fast_tokenizer.action_dim
            return WRAP("".join(map(ENCODE, action_tokens[0])))
        else:
            assert action.shape[1] == self.fast_tokenizer.time_horizon and \
                    action.shape[2] == self.fast_tokenizer.action_dim   
            return [WRAP("".join(map(ENCODE, da))) for da in action_tokens]

    def decode_token_ids_to_actions(self, action_token_ids: list[list[int]]) -> np.ndarray:
        min_fast_token = 0#self.fast_tokenizer.min_token
        max_fast_token = self.fast_tokenizer.vocab_size - 1 #self.fast_tokenizer.min_token + 
        fast_tokens = [
            # To make fast tokenizer happy using max/min to clip tokens into valid range
            [max(min(token_id - self.action_token_begin_idx, max_fast_token), min_fast_token) for token_id in batch]
            for batch in action_token_ids
        ]
        decoded_actions = self.fast_tokenizer.decode(fast_tokens).astype(np.float32) # return [batch, time_horizon, action_dim]
        return decoded_actions

    @property
    def vocab_size(self) -> int:
        return self.n_bins
