import json
import csv

# JSON 파일 읽기
with open('C:/270.인공지능기반 학생 진로탐색을 위한 상담 데이터 구축/01-1.정식개방데이터/Training/01.원천데이터/상담기록_데이터_고등학교.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# CSV 파일 쓰기
with open('output.csv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)

    # CSV 헤더 작성
    writer.writerow(
        ['student_idx', 'counseling_idx', 'counsellor_idx', 'counselling_purpose', 'counselling_satisfaction',
         'counselling_date', 'conv_idx', 'conv_category', 'self_eval', 'speaker_idx', 'utterance_idx',
         'utterance_delaytime', 'utterance'])

    # 각 대화에 대해 반복
    for conv in data.values():
        meta = conv['meta']
        conversation = conv['conversation']

        # 대화 정보 추출
        student_idx = meta['student_idx']
        counseling_idx = meta['counseling_idx']
        counsellor_idx = meta['counsellor_idx']
        counselling_purpose = meta['counselling_purpose']
        counselling_satisfaction = meta['counselling_satisfaction']
        counselling_date = meta['counselling_date']
        conv_idx = conversation[0]['conv_idx']
        conv_category = conversation[0]['conv_category']
        self_eval = conversation[0]['self_eval']

        # 발화 정보 추출
        utterances = [utterance['utterance'] for utterance in conversation[0]['utterances']]
        speaker_idx = ','.join([utterance['speaker_idx'] for utterance in conversation[0]['utterances']])
        utterance_idx = ','.join([str(utterance['utterance_idx']) for utterance in conversation[0]['utterances']])
        utterance_delaytime = ','.join(
            [str(utterance['utterance_delaytime']) for utterance in conversation[0]['utterances']])

        # CSV 파일에 쓰기
        writer.writerow([
            student_idx,
            counseling_idx,
            counsellor_idx,
            counselling_purpose,
            counselling_satisfaction,
            counselling_date,
            conv_idx,
            conv_category,
            ','.join(map(str, self_eval)),
            speaker_idx,
            utterance_idx,
            utterance_delaytime,
            ','.join(utterances)
        ])