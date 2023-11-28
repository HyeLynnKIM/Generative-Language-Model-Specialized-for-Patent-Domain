# Generative-Language-Model-Specialized-for-Patent-Domain
✏polyglot-ko를 활용한 특허 도메인 특화 사전학습 생성형 모델

## 프로젝트 설명
polyglot-go-1.3b을 기반으로 사전학습 데이터인 특허(키프리스)와 미세조정 데이터인 지식IN 데이터를 수집한 후, 사전학습 및 LoRA를 활용한 미세조정으로 특허 도메인에 특화된 챗봇 모델 구현
(%참고 학습 환경: A100 1대/256G)

## 코드설명
- kiriscrawling.py : 키프리스 사이트 크롤링을 위한 코드
- pdfparser.py : 키르리스에서 pdf로 된 내용을 추가로 크롤링하기 위한 코드
- TXTtoJson.py : 크롤링한 데이터를 원하는 형식으로 변경
- no-pretrain-fine.py : 기본 모델로 인퍼런스
- only_pretrain.py : 키프리스 데이터로 사전학습 후 인퍼런스
- only_fine.py : 사전학습 X 미세조정 후 인퍼런스
- pretrain-fine.py : 사전학습 O 미세조정 O 인퍼런스
- pretrain_code.py : 사전학습 코드
