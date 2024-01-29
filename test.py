import json
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification  # GPT-2 모델의 토크나이저를 사용하겠습니다.
from transformers import GPT2LMHeadModel, GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer, Trainer, TrainingArguments
import torch


class CounselingDataset(Dataset):
    def __init__(self, json_file_path, tokenizer, max_length=512):
        self.data = []
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for key, value in data.items():
                conversation = [utterance['utterance'] for session in value['conversation'] for utterance in
                                session['utterances']]
                self.data.append(' '.join(conversation))

        # 패딩 토큰 추가
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors='pt',
                                  padding='max_length')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


# 모델 초기화
model_name = "gpt2"  # 사용할 모델의 이름
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2ForSequenceClassification.from_pretrained(model_name)

# 데이터셋 설정
json_file_path = "C:\python\상담기록_데이터_초등학교.json"
dataset = CounselingDataset(json_file_path, tokenizer)

# 데이터로더 설정
batch_size = 4  # 적절한 배치 크기를 선택하세요.
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 모델 학습을 위한 Trainer 및 TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    save_steps=100,
    save_total_limit=3,
    logging_dir="./logs",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# 모델 학습
trainer.train()

# 학습이 끝난 모델을 저장하고 싶다면
model.save_pretrained("C:\python\saved_model")
tokenizer.save_pretrained("C:\python\saved_model")
