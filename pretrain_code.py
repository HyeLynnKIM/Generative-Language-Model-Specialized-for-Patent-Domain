import torch, os, json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

ds = load_dataset("nlpai-lab/kullm-v2", split="train")
total_ds = len(ds)
cnt = 0

total_path = 'total_data.txt'
path = 'pretrain_/'
data_list = next(os.walk(path))[1]

ff = open(f"{total_path}", 'w', encoding='utf-8')

tot = 0
for d in data_list:
    data_list2 = next(os.walk(path + d))[2]
    for d2 in data_list2:
        with open(f"{path}{d}/{d2}", encoding='utf-8') as f:
            data = json.load(f)
            ff.write(data['발명(고안)의 국문명칭'][1:-1] + '\n')
            ff.write(data['발명(고안)의 영문명칭'][1:-1] + '\n')

            anno = ''
            dd = data['요약서'][2:-2].split("'")

            for ddd in dd:
                if ddd != ', ':
                    anno += ddd
            ff.write(f'{anno}' + '\n')
            tot += 3
        if cnt < total_ds:
            ff.write(ds[cnt]['output'])
            cnt += 1
            ff.write(ds[cnt]['output'])
            cnt += 1
            tot += 2

print('end!!!', tot)
# exit(0)

# 모델 및 토크나이저 불러오기
model_name = "EleutherAI/polyglot-ko-1.3b"  # 또는 다른 GPT-2 모델 사이즈
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 데이터 로드 및 전처리
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="total_data.txt",  # 학습 데이터 파일 경로
    block_size=64  # 텍스트 블록 크기
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# 학습 설정 및 Trainer 생성
training_args = TrainingArguments(
    output_dir="poly-pretrained",  # 모델 및 결과물 저장 디렉토리
    overwrite_output_dir=True,
    num_train_epochs=2,  # 학습 에폭 수
    per_device_train_batch_size=32,  # 배치 크기
    save_steps=10_000,  # 주기적으로 모델 저장
    save_total_limit=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# 모델 사전 학습
trainer.train()

# 학습된 모델 저장
trainer.save_model()