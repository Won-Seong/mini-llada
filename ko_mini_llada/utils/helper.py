from transformers import AutoTokenizer, PreTrainedModel

def setup_chat_format(tokenizer: AutoTokenizer, model: PreTrainedModel):
    """
    Set up the tokenizer and model for chat-based interactions by adding special tokens
    """
    # 1. define special tokens
    special_tokens_dict = {
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]
    }
    
    # 2. add special tokens to tokenizer
    # num_new_tokens: the number of tokens actually added
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    # 3. Chat Template
    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n{% endif %}"
            "{% if message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% endif %}"
            "{% if message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )

    # 4. Resize model embeddings if new tokens were added
    if num_new_tokens > 0:
        print(f"Adding {num_new_tokens} new tokens to model embeddings.")
        model.resize_token_embeddings(len(tokenizer))
        
    return tokenizer, model