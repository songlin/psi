from typing import List, Union, Sequence, Any
import numpy as np
from transformers import PreTrainedTokenizerBase

TOK_ACTION_START = "<|action_start|>"
TOK_ACTION_END = "<|action_end|>"


class TextActionTokenizer:
    """
    Text-based action tokenizer.

    Encodes continuous actions by flattening them and joining numeric values
    into a space-separated string. Decoding performs the inverse operation.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        time_horizon : int,
        action_dim : int,
        scale: float = 1000.0,
    ) -> None:
        special_tokens_dict = {
            "additional_special_tokens": [TOK_ACTION_START, TOK_ACTION_END]
        }
        tokenizer.add_special_tokens(special_tokens_dict)  # type: ignore

        self.tokenizer = tokenizer
        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.scale = scale

        self.action_token_begin_idx = tokenizer(TOK_ACTION_START)["input_ids"][0]
        self.action_token_end_idx = tokenizer(TOK_ACTION_END)["input_ids"][0]

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
    def __call__(self, action: np.ndarray, wrap_special_tokens: bool = True) -> Union[str, List[str]]:

        def wrap(text: str) -> str:
            if wrap_special_tokens:
                return f"{TOK_ACTION_START}{text}{TOK_ACTION_END}"
            return text

        if action.ndim not in (2, 3):
            raise ValueError(f"Action must be 2D or 3D, got shape {action.shape}")

        def flatten_and_format(arr: np.ndarray) -> str:
            flat = arr.reshape(-1)
            # return ",".join(f"{x:.2f}" for x in flat)
            return ",".join(f"{int(x*self.scale)}" for x in flat)

        if action.ndim == 2:
            return wrap(flatten_and_format(action))

        outputs: List[str] = []
        for chunk in action:
            outputs.append(wrap(flatten_and_format(chunk)))

        return outputs

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    def decode_token_ids_to_actions(
        self,
        action_token_ids: Sequence,
    ) -> np.ndarray:
        """
        Decode token ids back into actions.

        Args:
            action_token_ids:
                List[int] for a single action, or
                List[List[int]] for a batch

        Returns:
            NumPy array representing decoded actions
        """

        def decode_one(token_ids: Sequence[int]) -> np.ndarray:
            text = self.tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            # print(text)

            if TOK_ACTION_START in text:
                text = text.replace(TOK_ACTION_START, "")
            if TOK_ACTION_END in text:
                text = text.replace(TOK_ACTION_END, "")
            ##hacking... TODO: fix this
            if "<|im_start|>" in text:
                text = text.replace("<|im_start|>", "")
            if "<|box_end|>" in text:
                text = text.replace("<|box_end|>", "")
            if "-<|action_end|>" in text:
                text = text.replace("-<|action_end|>", "")
            ##....
            text = text.strip()

            if not text:
                return np.empty((0,), dtype=np.float32)

            values = np.array(
                [float(x)/self.scale for x in text.split(",") if x != ""],
                dtype=np.float32,
            )

            # Reshape only if action_dim is an integer
            if isinstance(self.action_dim, int):
                if values.size % self.action_dim != 0:
                    raise ValueError(
                        f"Cannot reshape array of length {values.size} "
                        f"with action_dim={self.action_dim}"
                    )

                T = (
                    self.time_horizon
                    if isinstance(self.time_horizon, int)
                    else values.size // self.action_dim
                )
                values = values.reshape(T, self.action_dim)

            return values

        if not isinstance(action_token_ids, (list, tuple)):
            raise TypeError("action_token_ids must be a list or tuple")

        if len(action_token_ids) == 0:
            return np.empty((0,), dtype=np.float32)

        # Single action: List[int]
        if not isinstance(action_token_ids[0], (list, tuple)):
            return decode_one(action_token_ids)

        # Batch: List[List[int]]
        # decoded = [decode_one(ids) for ids in action_token_ids]
        decoded = []
        for i, ids in enumerate(action_token_ids):
            try:
                decoded.append(decode_one(ids)) 
            except ValueError as e:
                # print(
                #     f"[Action Decode Error] sample {i}, "
                #     f"error={e}"
                # )
                decoded.append(np.zeros((self.time_horizon, self.action_dim), dtype=np.float32))
        return np.stack(decoded, axis=0)