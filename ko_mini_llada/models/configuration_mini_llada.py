from transformers import PretrainedConfig, AutoConfig

class MiniLLaDAConfig(PretrainedConfig):
    model_type = "mini-llada"

    def __init__(
        self,
        vocab_size=32000,
        mask_token_id=None,
        dim=512,
        depth=12,
        head=16,
        intermediate_size=1024,
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