import torch
from datasets import load_dataset, concatenate_datasets

def prepare_dataset(
    tokenizer, 
    dataset_config: list[dict], 
    max_seq_len: int = 512, 
    mode: str = "pretrain",
):
    print(f"Preparing dataset for [{mode}] with max_len={max_seq_len}...")
    processed_datasets = []

    for config in dataset_config:
        name = config['name']
        subset = config.get('subset', None)
        split = config.get('split', 'train')
        limit = config.get('limit', None)
        
        print(f"Loading dataset: {name} (subset: {subset})...")

        try:
            # 1. Load Dataset
            if subset:
                dataset = load_dataset(name, subset, split=split)
            else:
                dataset = load_dataset(name, split=split)

            if limit:
                dataset = dataset.select(range(limit))

            # 2. Format Text (핵심 변경 구간)
            if mode == "sft":
                # [SFT] Chat Template 적용
                q_col = config.get('q_col', 'question')
                a_col = config.get('a_col', 'answer')
                
                def format_sft(example):
                    # 1. 데이터를 Chat Message 포맷(List[Dict])으로 변환
                    messages = [
                        {"role": "user", "content": example[q_col]},
                        {"role": "assistant", "content": example[a_col]}
                    ]
                    # 2. 토크나이저의 템플릿 적용 (문자열로 반환됨)
                    # tokenize=False로 해야 텍스트 상태로 합쳐집니다.
                    formatted_text = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=False
                    )
                    return {'text': formatted_text}
                
                dataset = dataset.map(format_sft, remove_columns=dataset.column_names)
                
            else: # mode == "pretrain"
                # [Pre-train] 기존 방식 유지 (Raw Text)
                text_col = config.get('text_col', 'text')
                
                def format_pretrain(example):
                    return {'text': example[text_col]}
                
                dataset = dataset.map(format_pretrain, remove_columns=dataset.column_names)

            # 공통: 너무 짧은 데이터 제거
            dataset = dataset.filter(lambda x: x['text'] is not None and len(x['text']) > 5)
            processed_datasets.append(dataset)

        except Exception as e:
            print(f"⚠️ Error loading dataset {name}: {e}")
            continue

    if not processed_datasets:
        raise ValueError("No datasets were successfully loaded.")
    
    # 3. Combine
    combined_dataset = concatenate_datasets(processed_datasets)

    # 4. Tokenization (Standard Padding)
    print("Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length",
            truncation=True, 
            max_length=max_seq_len, 
            return_tensors="pt"
        )

    # map 함수 적용
    tokenized_dataset = combined_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )

    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    print(f"DONE. Total samples: {len(tokenized_dataset)}")
    return tokenized_dataset