import json
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from psi.utils.overwatch import initialize_overwatch
overwatch = initialize_overwatch(__name__)
 
TOK_ACTION_START = "<|action_start|>"
TOK_ACTION_END = "<|action_end|>"


ENCODE = lambda x: f"<|a_{x}|>"


class VQActionTokenizer:
    """Loads a torch model (VqVaE) that turns"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        vq_vae_path="",
        device="cpu",
    ):
        self.tokenizer = tokenizer
        self.device = device

        ### VQ VAE loading ###

        # NOTE: if this errors, you need to install vqvae, source: https://github.com/jayLEE0301/vq_bet_official
        from vqvae.vqvae import VqVae

        self.vq_path = Path(vq_vae_path)
        assert self.vq_path.exists(), f"Missing VQ VAE path: {self.vq_path}"
        vq_model_path = self.vq_path / "checkpoints" / "model.pt"
        vq_config_path = self.vq_path / "config.json"
        assert vq_model_path.exists(), f"Missing VQ checkpoint path: {vq_model_path}"
        assert vq_config_path.exists(), f"Missing VQ config path: {vq_config_path}"
        with open(vq_config_path, "r") as f:
            vq_config = dict(json.load(f))
        # set the load checkpoint
        vq_config["load_dir"] = vq_model_path
        vq_config["eval"] = True
        vq_config["device"] = self.device
        overwatch.info(f"Loading VQ VAE for Action Tokenization from {vq_config_path}...")
        # instantiate the vqvae and load
        self.vq_vae = VqVae(**vq_config)
        overwatch.info(f"Found VQ VAE parameters: \n{self.vq_vae}")
        ### TOKENIZATION arguments ###
        # number of bins to assign for each "action" dimension
        self.n_bins = self.vq_vae.vqvae_n_embed

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

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        # make sure shape matches (1 x T x A)
        action = torch.from_numpy(action).to(self.device).reshape((1, self.vq_vae.input_dim_h, self.vq_vae.input_dim_w))
        # action is (1 x T x A), codes will be (1 x GROUPS) each between 0 and BINS-1
        _, vq_code = self.vq_vae.get_code(action)
        assert torch.all(vq_code >= 0) and torch.all(vq_code < self.n_bins)

        # vq_codes will be between [0, n_bins-1], so we subtract them from vocab_size - 1
        # for example, code 0 maps to vocab_size - 1
        return self.tokenizer.decode(list(self.action_token_begin_idx + self.n_bins -1 - vq_code[0].numpy()))

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        # first convert from tokens to bins (inverse of what happens in __call__)
        action_token_ids = self.action_token_begin_idx + self.n_bins -1 - action_token_ids
        initial_shape = action_token_ids.shape
        # these directly correspond to the bins
        action_token_ids = np.clip(action_token_ids, 0, self.n_bins - 1)
        action_token_ids = torch.from_numpy(action_token_ids).to(self.device).reshape(-1, self.vq_vae.vqvae_groups)
        assert torch.all(action_token_ids >= 0) and torch.all(action_token_ids < self.n_bins)
        # (1 x G) --> (1 x Z_DIM)
        latent = self.vq_vae.draw_code_forward(action_token_ids)
        # --> (1 x A) --> (A,)
        ret_action = self.vq_vae.get_action_from_latent(latent)

        # reshape to be a flat array if the input was a single action
        if action_token_ids.shape[0] == 1 and len(initial_shape) == 1:
            return ret_action[0]

        # get the first horizon element of the returned actions (VQ might return an action horizon)
        # TODO parameterize this
        return ret_action[:]

    @property
    def required_future_horizon(self) -> int:
        # the number of future action horizon elements
        return self.vq_vae.input_dim_h - 1

    @property
    def vocab_size(self) -> int:
        return self.n_bins