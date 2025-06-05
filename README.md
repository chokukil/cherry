# Cherry AI

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3.23+-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3.25+-orange.svg)
[![Open in - LangGraph Studio](https://img.shields.io/badge/Open_in-LangGraph_Studio-00324d.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4NS4zMzMiIGhlaWdodD0iODUuMzMzIiB2ZXJzaW9uPSIxLjAiIHZpZXdCb3g9IjAgMCA2NCA2NCI+PHBhdGggZD0iTTEzIDcuOGMtNi4zIDMuMS03LjEgNi4zLTYuOCAyNS43LjQgMjQuNi4zIDI0LjUgMjUuOSAyNC41QzU3LjUgNTggNTggNTcuNSA1OCAzMi4zIDU4IDcuMyA1Ni43IDYgMzIgNmMtMTIuOCAwLTE2LjEuMy0xOSAxLjhtMzcuNiAxNi42YzIuOCAyLjggMy40IDQuMiAzLjQgNy42cy0uNiA0LjgtMy40IDcuNkw0Ny4yIDQzSDE2LjhsLTMuNC0zLjRjLTQuOC00LjgtNC44LTEwLjQgMC0xNS4ybDMuNC0zLjRoMzAuNHoiLz48cGF0aCBkPSJNMTguOSAyNS42Yy0xLjEgMS4zLTEgMS43LjQgMi41LjkuNiAxLjcgMS44IDEuNyAyLjcgMCAxIC43IDIuOCAxLjYgNC4xIDEuNCAxLjkgMS40IDIuNS4zIDMuMi0xIC42LS42LjkgMS40LjkgMS41IDAgMi43LS41IDIuNy0xIDAtLjYgMS4xLS44IDIuNi0uNGwyLjYuNy0xLjgtMi45Yy01LjktOS4zLTkuNC0xMi4zLTExLjUtOS44TTM5IDI2YzAgMS4xLS45IDIuNS0yIDMuMi0yLjQgMS41LTIuNiAzLjQtLjUgNC4yLjguMyAyIDEuNyAyLjUgMy4xLjYgMS41IDEuNCAyLjMgMiAyIDEuNS0uOSAxLjItMy41LS40LTMuNS0yLjEgMC0yLjgtMi44LS44LTMuMyAxLjYtLjQgMS42LS41IDAtLjYtMS4xLS4xLTEuNS0uNi0xLjItMS42LjctMS43IDMuMy0yLjEgMy41LS41LjEuNS4yIDEuNi4zIDIuMiAwIC43LjkgMS40IDEuOSAxLjYgMi4xLjQgMi4zLTIuMy4yLTMuMi0uOC0uMy0yLTEuNy0yLjUtMy4xLTEuMS0zLTMtMy4zLTMtLjUiLz48L3N2Zz4=)](https://langgraph-studio.vercel.app/templates/open?githubUrl=https://github.com/langchain-ai/react-agent)

## 프로젝트 개요

> 🤖 **차세대 AI Data Scientist** 채팅 인터페이스

`Cherry AI`는 Model Context Protocol(MCP)을 통해 다양한 외부 도구와 데이터 소스에 접근할 수 있는 ReAct 에이전트를 구현한 프로젝트입니다. **특히 AI Data Scientist Agent**를 통해 전문가 수준의 데이터 분석과 머신러닝을 자동화합니다.

## 🚀 AI Data Scientist Agent 주요 기능

### 🤖 완전 자동화 AutoML 파이프라인
- **원클릭 AutoML**: `auto_ml_pipeline(df, 'target_col')` 한 줄로 완전 자동화
- **지능형 문제 유형 감지**: 분류/회귀/클러스터링/시계열을 자동으로 판단
- **데이터 크기별 최적 모델 선택**: 소규모부터 대용량 데이터까지 맞춤형 알고리즘 추천
- **자동 데이터 전처리**: 결측값, 범주형 데이터, 스케일링을 지능적으로 처리
- **하이퍼파라미터 자동 튜닝**: GridSearch/RandomSearch 자동 실행
- **모델 성능 자동 평가**: 교차검증 및 다양한 메트릭으로 객관적 평가
- **전문가 수준 분석 보고서**: 인사이트와 추천사항을 포함한 종합 보고서 생성

### 🧠 고급 머신러닝 모델 지원
- **전통적 ML**: scikit-learn 전체 모듈 (분류, 회귀, 클러스터링, 차원축소)
- **고급 부스팅**: XGBoost, LightGBM, CatBoost 지원
- **딥러닝**: TensorFlow/Keras 자동 신경망 모델 생성
- **시계열 분석**: ARIMA, 계절성 분해 등 statsmodels 지원
- **모델 해석**: SHAP 기반 예측 해석 및 피처 중요도 분석

### 📊 지능형 시각화 시스템
- **plt.show() 자동 변환**: matplotlib 시각화를 Streamlit에서 영구 보존
- **한글 폰트 자동 설정**: Windows/macOS/Linux 환경별 최적 한글 폰트 자동 적용
- **실시간 시각화**: 분석 과정에서 생성된 모든 차트를 실시간으로 표시
- **시각화 관리**: 생성된 차트들을 체계적으로 관리하고 재사용 가능

### 🔍 자동 데이터 탐색 (EDA)
- **CSV 업로드 시 자동 분석**: 데이터 구조, 타입, 결측값 등을 즉시 분석
- **ML 적합성 진단**: 데이터 특성에 따른 추천 알고리즘과 분석 전략 제시
- **맞춤형 분석 가이드**: 초보자부터 전문가까지 단계별 가이드 제공
- **후속 질문 자동 생성**: 분석 결과에 따른 맞춤형 후속 질문 제안

### 💬 사용자 친화적 인터페이스
- **대화형 분석**: 자연어로 질문하면 코드 생성 및 실행으로 답변
- **단계별 가이드**: 초보자(원클릭) → 중급자(커스텀) → 고급자(전문가) 모드
- **오류 복구 시스템**: 분석 중 오류 발생 시 자동 진단 및 대안 제시
- **실시간 피드백**: 분석 진행 상황을 실시간으로 확인 가능

## 🛠️ 동적 도구 관리 시스템
**MCP 도구 설정** : 사이드바에서 MCP 도구를 설정하여 동적으로 반영할 수 있습니다.
**시스템 프롬프트 설정** : 사이드바에서 프롬프트를 선택하여 동적으로 반영할 수 있습니다.

## 🎯 사용 시나리오

### 📈 비즈니스 데이터 분석
```python
# 매출 데이터 분석 예시
results = auto_ml_pipeline(sales_df, target_col='revenue')
print(generate_ml_report(results))
```

### 🔬 과학 연구 데이터
```python
# 실험 데이터 클러스터링 분석
results = auto_ml_pipeline(experiment_df)  # 비지도학습
```

### 💰 금융 예측 모델
```python
# 주가 예측 딥러닝 모델
dl_model = create_deep_learning_model('regression', input_shape=X.shape[1])
```

### 📊 마케팅 고객 분석
```python
# 고객 세분화 및 해석
results = auto_ml_pipeline(customer_df, target_col='segment')
shap_values = explain_model_predictions(results['best_model']['model'], X_test)
```


# Windows 환경 설정 가이드
### 1. uv 설치
- 공식 사이트에서 직접 다운로드: https://docs.astral.sh/uv/getting-started/installation/
- 또는 pip으로 설치: `pip install uv`

### GPU 패키지 설치 실패
1. NVIDIA 드라이버 최신 버전 확인
2. CUDA 호환성 확인 (TensorFlow 2.15는 CUDA 12.x 지원)
3. CPU 버전으로 대체 설치

### 메모리 부족 오류
- 가상 메모리 늘리기
- 불필요한 프로그램 종료 후 재시도
- 개별 패키지로 나누어 설치

## 🎯 설치 완료 후 사용법

### 1. Streamlit 앱 실행
```bash
uv venv
.venv\Scripts\activate
uv pip install -e .
streamlit run Home.py
```

### 2. MCP Server 실행
```bash
uv run mcp_file_management.py
uv run mcp_private_rag.py
```

### 3. AI Data Scientist 사용하기
1. **CSV 파일 업로드**: 사이드바에서 데이터 파일 업로드
2. **Agent 생성**: 'Agent 생성하기' 버튼 클릭
3. **자동 분석 시작**: '🚀 자동 데이터 분석 시작' 버튼 클릭
4. **대화형 분석**: 자연어로 추가 분석 요청

## 💡 팁

1. **설치 전 시스템 요구사항 확인**
   - Windows 10/11 (64비트)
   - Python 3.11+ 
   - 여유 공간 5GB 이상
   - NVIDIA GPU (GPU 버전 시)

2. **성능 최적화**
   - SSD 사용 권장
   - 메모리 8GB 이상 권장
   - GPU 메모리 4GB 이상 권장

3. **데이터 분석 성능 최적화**
   - 대용량 데이터(100K+ 행): LightGBM, 샘플링 기법 자동 적용
   - 소규모 데이터(1K 미만): 교차검증 강화, 정규화 자동 적용
   - 고차원 데이터: PCA, 피처 선택 기법 자동 추천

4. **백업 및 복원**
   - `uv export > requirements.txt` - 현재 환경 백업
   - `uv sync` - 백업된 환경 복원
