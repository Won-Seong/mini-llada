import os
import yaml
import argparse
import torch
from transformers import (
    Trainer, 
    TrainingArguments, 
    AutoTokenizer, 
    DataCollatorWithPadding
)
from datasets import load_dataset

# 사용자님이 만드신 모듈 임포트
from ko_mini_llada.models.configuration_ko_mini_llada import LladaConfig
from ko_mini_llada.models.modeling_ko_mini_llada import KoMiniLlada
from ko_mini_llada.data.dataset import prepare_dataset

# (선택) 이전에 만든 생성 평가 콜백이 있다면 임포트
from ko_mini_llada.utils.callbacks import GenerateSampleCallback 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/config.yaml")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--mode", type=str, default="pretrain", choices=["pretrain", "sft"])
    parser.add_argument("--pad_with_eos", action="store_true")
    args_cli = parser.parse_args()

    # 1. Config 파일 로드
    with open(args_cli.config_file, "r") as f:
        config = yaml.safe_load(f)

    # 2. Tokenizer & Model 초기화
    # (이미 Hub에 올린 모델이 있다면 AutoModel.from_pretrained("YourID/...")로 로드 가능)
    # 여기서는 Config 기반 초기화 예시입니다.
    tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model_name'])
    
    # [중요] SFT 모드일 경우 Chat Template 설정 (필요시)
    if args_cli.mode == "sft":
        # ... (이전 대화의 Chat Template 설정 코드 추가) ...
        pass

    llada_config = LladaConfig(
        backbone_model_name=config['pretrained_model_name'],
        mask_token_id=tokenizer.mask_token_id
    )
    
    model = KoMiniLlada(llada_config)
    
    # Special Token이 추가되었다면 임베딩 리사이즈
    model.resize_token_embeddings(len(tokenizer))

    # 3. 데이터셋 준비
    # 기존 prepare_dataset 함수 활용
    full_dataset = prepare_dataset(
        tokenizer, 
        dataset_config=config['dataset_config']['pre_training' if args_cli.mode == 'pretrain' else 'fine_tuning']['dataset_list'], 
        max_seq_len=config['max_seq_len'],
        mode=args_cli.mode,
        pad_with_eos=args_cli.pad_with_eos
    )

    # [핵심 1] Train/Eval Split
    # Config에 있는 test_size 사용
    test_size = config['dataset_config'].get('pre_training' if args_cli.mode == 'pretrain' else 'fine_tuning').get('test_size', 0.01)
    split_datasets = full_dataset.train_test_split(test_size=test_size, seed=config.get('random_seed', 42))
    
    train_dataset = split_datasets['train']
    eval_dataset = split_datasets['test']

    # [핵심 2] 'labels' 컬럼 생성
    # HF Trainer는 데이터셋에 'labels'가 있어야 학습 모드(Loss 계산)로 진입합니다.
    # KoMiniLlada는 input_ids를 복사해서 labels로 주면 내부에서 마스킹하고 Loss를 계산합니다.
    print("Mapping labels...")
    train_dataset = train_dataset.map(lambda x: {'labels': x['input_ids']})
    eval_dataset = eval_dataset.map(lambda x: {'labels': x['input_ids']})

    # 4. TrainingArguments 설정 (config.yaml의 내용 매핑)
    train_conf = config['train_config']
    
    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        overwrite_output_dir=True,
        
        # 학습 파라미터
        num_train_epochs=train_conf.get('num_epochs', 3),
        per_device_train_batch_size=train_conf.get('batch_size', 8),
        per_device_eval_batch_size=train_conf.get('batch_size', 8),
        gradient_accumulation_steps=train_conf.get('gradient_accumulation_steps', 1),
        learning_rate=float(train_conf.get('learning_rate', 5e-5)),
        weight_decay=0.01,
        
        # 평가 및 저장 전략
        evaluation_strategy="steps",
        eval_steps=train_conf.get('eval_steps', 1000),
        save_strategy="steps",
        save_steps=train_conf.get('eval_steps', 1000),
        save_total_limit=2, # 체크포인트 개수 제한 (용량 관리)
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        
        # 하드웨어 및 효율성
        fp16=torch.cuda.is_available(), # GPU 있으면 fp16 자동 사용
        dataloader_num_workers=train_conf.get('num_workers', 4),
        
        # [중요] 커스텀 모델 사용 시 필수 옵션
        # Trainer가 알지 못하는 컬럼(labels 등)을 자동으로 지우지 않도록 설정
        remove_unused_columns=False, 
        
        # 로깅
        logging_steps=100,
        report_to="none", # wandb 등을 쓴다면 "wandb"
        run_name="mini-llada-run"
    )

    # 5. Trainer 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer), # Dynamic Padding 지원
        callbacks=[GenerateSampleCallback(tokenizer)] # (선택사항) 생성 결과 확인용
    )

    # 6. 학습 시작
    print("🚀 Start Training...")
    trainer.train()

    # 7. 최종 모델 저장
    print(f"💾 Saving final model to {args_cli.output_dir}/final")
    trainer.save_model(os.path.join(args_cli.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args_cli.output_dir, "final"))

if __name__ == "__main__":
    main()