# Deep Learning Projects

**딥러닝 이론부터 TensorFlow/Keras 기반의 다양한 프로젝트** 학습

---

##  Repository Structure
 deep-learning
├─ DL theory/
│ ├─ ML.txt # 머신러닝 이론 정리
│ ├─ deep_learning.py # 딥러닝 기본 코드
│ └─ tensor.py # Tensor 연산 예제
│
├─ Project 1/ # 대학원 합격 확률 예측
│ ├─ School probability prediction.py
│ └─ gpascore.csv # GPA & 점수 데이터
│
├─ Project 2/ # CNN 이미지 분류
│ ├─ Cat and Dog Image Classification practice.py
│ ├─ Cat and Dog Image Classification.py
│ ├─ Fashion Clothing Image Classification.py
│ ├─ Functional API.py
│ ├─ Image Augmentation.py
│ ├─ inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
│ ├─ kaggle.json
│ ├─ model save.py
│ ├─ model1.keras
│ ├─ sample_submission.csv
│ ├─ tensor board.py
│ └─ transfer learning & fine tuning.py
│
├─ Project 3,4/ # RNN, LSTM, GRU & NLP
│ ├─ Cleanbot.py # 악플 검사 AI
│ ├─ Composition AI.py # 작곡 AI v1
│ ├─ composition AI2.py # 작곡 AI v2
│ ├─ LSTM, GRU.txt # RNN/LSTM/GRU 이론
│ ├─ model1.keras
│ ├─ naver_shopping.txt # 한글 리뷰 데이터
│ └─ pianoabc.txt # 작곡 데이터셋
│
├─ Project 5/ # CSV 데이터 분석
│ ├─ Probability of dying.py
│ ├─ test.csv
│ └─ train.csv
│
├─ Project 6/ # GAN (Generative Adversarial Network)
│ ├─ GAN.py
│ └─ gan_img/ # GAN 생성 이미지
│
├─ .gitignore
└─ README.md

---

##  프로젝트 개요

### **Project 1: 대학원 합격 확률 예측**
- CSV 데이터를 활용해 합격 확률을 예측하는 모델
- **기술**: Pandas, TensorFlow, Regression/Classification  

### **Project 2: CNN 기반 이미지 분류**
- 패션 의류 이미지 분류 (Fashion-MNIST)
- 개/고양이 이미지 분류 (Kaggle 데이터셋)
- Functional API, 이미지 증강, TensorBoard, 전이학습(InceptionV3) 적용  

### **Project 3 & 4: RNN, LSTM, GRU & NLP**
- **작곡 AI**: LSTM을 활용한 음악 생성  
- **악플 검사 AI**: 한글 텍스트 전처리 & 감정 분석 (네이버 쇼핑 리뷰 데이터)  

### **Project 5: CSV 데이터 분석**
- Titanic과 유사한 데이터셋으로 생존 확률 예측  
- 전처리 → 모델링 → 예측 파이프라인 구현  

### **Project 6: GAN (Generative Adversarial Network)**
- GAN 기본 구현 및 학습  
- 사람 얼굴 이미지 생성 (Generator/Discriminator 모델링) 
