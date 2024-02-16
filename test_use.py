import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import re

# 토크나이저와 모델을 로드할 경로 설정
tokenizer_path = 'gogamza/kobart-base-v2'  # 토크나이저 저장된 디렉토리 경로
model_path = 'C:/python/counseling.pth'  # 모델 파일 경로

try:
    # 토크나이저 로드
    tokenizer = BartTokenizer.from_pretrained(tokenizer_path)

    # 모델 아키텍처 로드
    model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2")

    # 모델의 가중치 로드
    model.load_state_dict(torch.load(model_path))

    # 모델을 평가 모드로 설정
    model.eval()

except Exception as e:
    print(f"An error occurred while loading the model and tokenizer: {str(e)}")


# 대화 생성 함수
def generate_response(input_text):
    # 정규식을 사용하여 특수 문자 및 이상한 문자를 제거하거나 대체
    input_text = re.sub(r'[^\w\s]', '', input_text)
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


# 사용자 입력 받고 응답 생성
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = generate_response(user_input)
    print("Bot:", response)
