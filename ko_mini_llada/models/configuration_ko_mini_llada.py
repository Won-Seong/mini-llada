from transformers import PretrainedConfig

class LladaConfig(PretrainedConfig):
    model_type = "mini-llada"

    def __init__(
        self,
        backbone_model_name="klue/roberta-large",
        mask_token_id=None,
        max_seq_len=512,
        **kwargs,
    ):
        self.backbone_model_name = backbone_model_name
        self.mask_token_id = mask_token_id
        self.max_seq_len = max_seq_len
        super().__init__(**kwargs)