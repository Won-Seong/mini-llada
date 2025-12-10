from datasets import load_dataset, concatenate_datasets
from itertools import chain

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
    if mode == "pretrain":
        print("Tokenizing and Grouping (Chunking) for Pretraining...")
        
        # 1) 일단 텍스트를 전부 ID로 변환 (Padding 없이)
        def tokenize_raw(examples):
            return tokenizer(examples["text"], truncation=False, return_attention_mask=False)
        
        tokenized_dataset = combined_dataset.map(
            tokenize_raw, 
            batched=True, 
            remove_columns=["text"]
        )

        # 2) 블록 단위로 자르기 (Grouping function)
        block_size = max_seq_len
        
        def group_texts(examples):
            # 배치 내의 모든 텍스트(input_ids)를 하나로 이어 붙임
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            
            # 나머지가 생기면 버릴지(drop), 패딩할지 결정.
            # 여기서는 "마지막 문장만 padding" 요청하셨으므로,
            # total_length가 block_size보다 크거나 같을 때만 자릅니다.
            
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            
            # block_size 단위로 뚝뚝 끊어서 새로운 샘플 생성
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            
            # [옵션] 마지막 자투리 처리 (Padding)
            # 위 로직은 나머지를 버리는 방식(Hugging Face 예제 표준)입니다.
            result["labels"] = result["input_ids"].copy()
            return result

        lm_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            batch_size=1000, 
        )
        
        print(f"DONE. Chunked samples: {len(lm_dataset)}")
        return lm_dataset

    else:
        # [SFT] 기존 방식 (Padding & Truncation per sample)
        print("Tokenizing for SFT (Padding per sample)...")
        def tokenize_sft(examples):
            return tokenizer(
                examples["text"], 
                padding="max_length",
                truncation=True, 
                max_length=max_seq_len, 
                return_tensors="pt"
            )
        
        tokenized_dataset = combined_dataset.map(
            tokenize_sft, 
            batched=True, 
            remove_columns=["text"]
        )
        print(f"DONE. SFT samples: {len(tokenized_dataset)}")
        return tokenized_dataset