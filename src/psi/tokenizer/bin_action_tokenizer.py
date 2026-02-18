"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase

TOK_ACTION_START = "<|action_start|>"
TOK_ACTION_END = "<|action_end|>"


ENCODE = lambda x: f"<|a_{x}|>"


class BinActionTokenizer:
    
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_action: float = -1,
        max_action: float = 1,
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        NOTE =>> Songlin: (Sep 24, 2025)
        The default implementation is only suitable for tokenizers that can do round-trip tokenization and
        detokenization of arbitrary tokens. This is true for sentencepiece-based tokenizers like LlamaTokenizer, but
        not for tokenizers like Qwen2Tokenizer. e.g., tokenizer.encode(tokenizer.decode(151638))[0] != 151638.


        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = (
            tokenizer,
            bins,
            min_action,
            max_action,
        )

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Expand the tokenizer's vocabulary with new action tokens

        new_tokens = []
        for i in range(self.n_bins):
            new_tokens.append(ENCODE(i))
        tokenizer.add_tokens(new_tokens)

        special_tokens_dict = {
            "additional_special_tokens": [TOK_ACTION_START, TOK_ACTION_END]
        }
        tokenizer.add_special_tokens(special_tokens_dict)  # type: ignore

        self.token_id_to_bin_action = zip(
            range(self.n_bins), list(np.array(tokenizer(new_tokens)["input_ids"])[:, 0])
        )
        self.action_token_begin_idx = tokenizer(new_tokens[0])["input_ids"][0]

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        # self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))  # type: ignore

    def __call__(self, action: np.ndarray, wrap_special_tokens: bool = False) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        action = np.clip(
            action, a_min=float(self.min_action), a_max=float(self.max_action)
        )
        discretized_action = np.digitize(action, self.bins)

        WRAP = lambda x: f"{TOK_ACTION_START}{x}{TOK_ACTION_END}" if wrap_special_tokens else x

        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            # return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))  # type: ignore
            return WRAP("".join(map(ENCODE, list(discretized_action))))
        else:
            # return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())  # type: ignore
            return [WRAP("".join(map(ENCODE, da))) for da in list(discretized_action)]

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        """
        # discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = action_token_ids - self.action_token_begin_idx
        discretized_actions = np.clip(
            discretized_actions, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )

        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins
