import json
from torch.utils.data import Dataset


class CounselingDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        conversation = self.data[str(idx)]['conversation']

        # 대화의 모든 발화를 하나의 문장으로 결합
        context = ' '.join([utterance['utterance'] for turn in conversation for utterance in turn['utterances']])

        # 모델이 예측해야 할 대화의 마지막 발화
        response = conversation[-1]['utterances'][-1]['utterance']

        return {'context': context, 'response': response}


# 파일 경로 지정
file_path = 'C:\python\상담기록_데이터_초등학교.json'

# 데이터셋 생성
counseling_dataset = CounselingDataset(file_path)

# 데이터셋 예시 출력
for i in range(5):
    sample = counseling_dataset[i]
    print(f"Context: {sample['context']}")
    print(f"Response: {sample['response']}")
    print("=" * 50)
