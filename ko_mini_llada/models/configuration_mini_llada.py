from transformers import PretrainedConfig, AutoConfig

class MiniLLaDAConfig(PretrainedConfig):
    model_type = "mini-llada"

    def __init__(
        self,
        mask_token_id,
        vocab_size,
        **kwargs,
    ):
        # 1. get backbone config
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.dim = 512
        self.depth = 12
        self.head = 16
        self.intermediate_size = 1024
        self.max_seq_len = 2048
        
        super().__init__(**kwargs)