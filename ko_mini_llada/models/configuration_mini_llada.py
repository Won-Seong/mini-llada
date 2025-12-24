from transformers import PretrainedConfig, AutoConfig

class MiniLLaDAConfig(PretrainedConfig):
    model_type = "mini-llada"

    def __init__(
        self,
        vocab_size=52000,
        mask_token_id=4,
        dim=2048,
        depth=18,
        head=16,
        intermediate_size=5632,
        max_seq_len=2048,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.dim = dim
        self.depth = depth
        self.head = head
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        
        super().__init__(**kwargs)