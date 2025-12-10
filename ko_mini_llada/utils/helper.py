from transformers import AutoTokenizer

def setup_chat_format(tokenizer: AutoTokenizer):
    """
    Set up the tokenizer and model for chat-based interactions by adding special tokens
    """

    # 1. Chat Template
    chat_template = (
        "{{ bos_token }}"  # 시작할 때 [CLS]
        "{% for message in messages %}"
            # 1. User 인 경우
            "{% if message['role'] == 'user' %}"
                "### User:\n"
                "{{ message['content'] + eos_token }}"  # 내용 + [SEP]
            "{% endif %}"
            
            # 2. Assistant 인 경우
            "{% if message['role'] == 'assistant' %}"
                "### Assistant:\n"
                "{{ message['content'] + eos_token }}"  # 내용 + [SEP]
            "{% endif %}"
            
            # 3. System 인 경우 (필요하면)
            "{% if message['role'] == 'system' %}"
                "### System:\n"
                "{{ message['content'] + eos_token }}"
            "{% endif %}"
        "{% endfor %}"
        
        # 4. 생성 프롬프트 (Assistant가 말할 차례임을 알림)
        "{% if add_generation_prompt %}"
            "### Assistant:\n"
        "{% endif %}"
    )

    tokenizer.chat_template = chat_template
    return tokenizer