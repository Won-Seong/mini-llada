from transformers import PretrainedConfig, AutoConfig

class MiniLLaDAConfig(PretrainedConfig):
    model_type = "roberta"

    def __init__(
        self,
        backbone_model_name="klue/roberta-large",
        mask_token_id=4,
        **kwargs,
    ):
        # 1. get backbone config
        backbone_config = AutoConfig.from_pretrained(backbone_model_name)
        backbone_config_dict = backbone_config.to_dict()
        
        # 2. set specific attributes
        for key, value in backbone_config_dict.items():
            if key not in kwargs:
                kwargs[key] = value

        # 3. set the essential attributes
        self.backbone_config = backbone_config
        self.vocab_size = backbone_config.vocab_size
        self.mask_token_id = mask_token_id
        self.backbone_model_name = backbone_model_name

        super().__init__(**kwargs)