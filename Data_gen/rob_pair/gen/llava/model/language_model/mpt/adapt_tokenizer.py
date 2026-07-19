from typing import Union
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
NUM_SENTINEL_TOKENS: int = 100

def adapt_tokenizer_for_denoising(tokenizer: Tokenizer):

    sentinels_to_add = [f'<extra_id_{i}>' for i in range(NUM_SENTINEL_TOKENS)]
    tokenizer.add_tokens(sentinels_to_add, special_tokens=True)
    if tokenizer.pad_token is None:
        tokenizer.add_tokens('<pad>', special_tokens=True)
        tokenizer.pad_token = '<pad>'
        assert tokenizer.pad_token_id is not None
    sentinels = ''.join([f'<extra_id_{i}>' for i in range(NUM_SENTINEL_TOKENS)])
    _sentinel_token_ids = tokenizer(sentinels, add_special_tokens=False).input_ids
    tokenizer.sentinel_token_ids = _sentinel_token_ids

class AutoTokenizerForMOD(AutoTokenizer):


    @classmethod
    def from_pretrained(cls, *args, **kwargs):

        tokenizer = super().from_pretrained(*args, **kwargs)
        adapt_tokenizer_for_denoising(tokenizer)
        return tokenizer
