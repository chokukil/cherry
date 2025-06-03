<<<<<<< HEAD
# LangGraph Dynamic MCP Agents

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3.23+-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3.25+-orange.svg)
[![Open in - LangGraph Studio](https://img.shields.io/badge/Open_in-LangGraph_Studio-00324d.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4NS4zMzMiIGhlaWdodD0iODUuMzMzIiB2ZXJzaW9uPSIxLjAiIHZpZXdCb3g9IjAgMCA2NCA2NCI+PHBhdGggZD0iTTEzIDcuOGMtNi4zIDMuMS03LjEgNi4zLTYuOCAyNS43LjQgMjQuNi4zIDI0LjUgMjUuOSAyNC41QzU3LjUgNTggNTggNTcuNSA1OCAzMi4zIDU4IDcuMyA1Ni43IDYgMzIgNmMtMTIuOCAwLTE2LjEuMy0xOSAxLjhtMzcuNiAxNi42YzIuOCAyLjggMy40IDQuMiAzLjQgNy42cy0uNiA0LjgtMy40IDcuNkw0Ny4yIDQzSDE2LjhsLTMuNC0zLjRjLTQuOC00LjgtNC44LTEwLjQgMC0xNS4ybDMuNC0zLjRoMzAuNHoiLz48cGF0aCBkPSJNMTguOSAyNS42Yy0xLjEgMS4zLTEgMS43LjQgMi41LjkuNiAxLjcgMS44IDEuNyAyLjcgMCAxIC43IDIuOCAxLjYgNC4xIDEuNCAxLjkgMS40IDIuNS4zIDMuMi0xIC42LS42LjkgMS40LjkgMS41IDAgMi43LS41IDIuNy0xIDAtLjYgMS4xLS44IDIuNi0uNGwyLjYuNy0xLjgtMi45Yy01LjktOS4zLTkuNC0xMi4zLTExLjUtOS44TTM5IDI2YzAgMS4xLS45IDIuNS0yIDMuMi0yLjQgMS41LTIuNiAzLjQtLjUgNC4yLjguMyAyIDEuNyAyLjUgMy4xLjYgMS41IDEuNCAyLjMgMiAyIDEuNS0uOSAxLjItMy41LS40LTMuNS0yLjEgMC0yLjgtMi44LS44LTMuMyAxLjYtLjQgMS42LS41IDAtLjYtMS4xLS4xLTEuNS0uNi0xLjItMS42LjctMS43IDMuMy0yLjEgMy41LS41LjEuNS4yIDEuNi4zIDIuMiAwIC43LjkgMS40IDEuOSAxLjYgMi4xLjQgMi4zLTIuMy4yLTMuMi0uOC0uMy0yLTEuNy0yLjUtMy4xLTEuMS0zLTMtMy4zLTMtLjUiLz48L3N2Zz4=)](https://langgraph-studio.vercel.app/templates/open?githubUrl=https://github.com/langchain-ai/react-agent)

## 프로젝트 개요

> 채팅 인터페이스

![Project Overview](./assets/Project-Overview.png)

`Agentic AI`는 Model Context Protocol(MCP)을 통해 다양한 외부 도구와 데이터 소스에 접근할 수 있는 ReAct 에이전트를 구현한 프로젝트입니다. 이 프로젝트는 LangGraph 의 ReAct 에이전트를 기반으로 하며, MCP 도구를 쉽게 추가하고 구성할 수 있는 인터페이스를 제공합니다.

![Project Demo](./assets/MCP-Agents-TeddyFlow.png)

### 주요 기능
 
**동적 방식으로 도구 설정 대시보드**

`http://localhost:2025` 에 접속하여 도구 설정 대시보드를 확인할 수 있습니다.

![Tool Settings](./assets/Tools-Settings.png)

**도구 추가** 탭에서 [Smithery](https://smithery.io) 에서 사용할 MCP 도구의 JSON 구성을 복사 붙여넣기 하여 도구를 추가할 수 있습니다.

![Tool Settings](./assets/Add-Tools.png)

----

**실시간 반영**

도구 설정 대시보드에서 도구를 추가하거나 수정하면 실시간으로 반영됩니다.

![List Tools](./assets/List-Tools.png)

**시스템 프롬프트 설정**

`prompts/system_prompt.yaml` 파일을 수정하여 시스템 프롬프트를 설정할 수 있습니다.

이 또한 동적으로 바로 반영되는 형태입니다.

![System Prompt](./assets/System-Prompt.png)

만약, 에이전트에 설정되는 시스템프롬프트를 수정하고 싶다면 `prompts/system_prompt.yaml` 파일의 내용을 수정하면 됩니다.

----

### 주요 기능

* **LangGraph ReAct 에이전트**: LangGraph를 기반으로 하는 ReAct 에이전트
* **실시간 동적 도구 관리**: MCP 도구를 쉽게 추가, 제거, 구성 가능 (Smithery JSON 형식 지원)
* **실시간 동적 시스템 프롬프트 설정**: 시스템 프롬프트를 쉽게 수정 가능 (동적 반영)
* **대화 기록**: 에이전트와의 대화 내용 추적 및 관리
* **localhost 지원**: localhost 로 실행 가능(채팅 인터페이스 연동 가능)

# Windows 환경 설정 가이드

## 🚀 빠른 시작 (추천)

### 방법 1: PowerShell 스크립트 (고급 기능 포함)

1. **PowerShell을 관리자 권한으로 실행**
   ```
   Windows 키 + X → "Windows PowerShell (관리자)"
   ```

2. **실행 정책 설정 (필요시)**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **스크립트 다운로드 및 실행**
   ```powershell
   # 스크립트 파일을 프로젝트 폴더에 저장 후
   .\install_ai_environment.ps1
   
   # 또는 옵션과 함께 실행
   .\install_ai_environment.ps1 -Mode auto
   .\install_ai_environment.ps1 -Mode cpu
   .\install_ai_environment.ps1 -Mode gpu
   .\install_ai_environment.ps1 -Mode full
   ```

### 방법 2: Batch 파일 (단순하고 안정적)

1. **명령 프롬프트를 관리자 권한으로 실행**
   ```
   Windows 키 + X → "명령 프롬프트 (관리자)"
   ```

2. **스크립트 실행**
   ```batch
   install_ai_environment.bat
   ```

## 🛠️ 수동 설치 (문제 발생 시)

### 1. uv 설치

```powershell
# PowerShell에서 실행
Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile "install_uv.ps1"
PowerShell -ExecutionPolicy Bypass -File "install_uv.ps1"
Remove-Item "install_uv.ps1" -Force
```

### 2. 기본 패키지 설치

```bash
# 프로젝트 폴더에서 실행
uv sync
```

### 3. CPU 버전 ML/DL 패키지

```bash
uv add "tensorflow-cpu>=2.15.0"
uv add "torch>=2.0.0" "torchvision>=0.15.0" "torchaudio>=2.0.0"
uv add "xgboost>=2.0.0" "lightgbm>=4.0.0" "catboost>=1.2.0"
uv add "shap>=0.45.0" "statsmodels>=0.14.0"
uv add "plotly>=5.15.0" "wordcloud>=1.9.0" "openpyxl>=3.1.0"
uv add "nltk>=3.8.0" "Pillow>=10.0.0"
```

### 4. GPU 버전 (NVIDIA GPU 있는 경우)

```bash
uv add "tensorflow[and-cuda]>=2.15.0"
uv add "torch>=2.0.0" "torchvision>=0.15.0" "torchaudio>=2.0.0"
uv add "transformers>=4.30.0" "datasets>=2.12.0" "accelerate>=0.20.0"
# 나머지는 CPU 버전과 동일
```

## 🔍 설치 확인 방법

### 1. 기본 확인

```bash
python -c "import tensorflow as tf; import torch; print(f'TF: {tf.__version__}, PyTorch: {torch.__version__}')"
```

### 2. GPU 확인

```bash
python -c "
import tensorflow as tf
import torch
print(f'TF GPU: {len(tf.config.list_physical_devices(\"GPU\"))}개')
print(f'PyTorch CUDA: {torch.cuda.is_available()}, {torch.cuda.device_count()}개')
"
```

### 3. 환경 정보 확인

```bash
python ai_config.py
python gpu_utils.py
```

## ⚠️ 문제 해결

### PowerShell 실행 정책 오류
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### uv 설치 실패
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
streamlit run 03_Data_Agent.py
```

### 2. AutoML 사용 예시
```python
# CSV 파일 업로드 후
results = auto_ml_pipeline(df, target_col='target_column')
print(generate_ml_report(results))
```

### 3. GPU 최적화 (GPU 설치한 경우)
```python
from gpu_utils import optimize_for_environment
optimize_for_environment()
```

## 📊 설치 옵션별 특징

| 옵션 | 용량 | 설치 시간 | GPU 지원 | 추천 대상 |
|------|------|----------|----------|-----------|
| **CPU** | ~2GB | 5-10분 | ❌ | 일반 사용자, 안정성 중시 |
| **GPU** | ~4GB | 10-20분 | ✅ | NVIDIA GPU 보유자 |
| **Full** | ~6GB | 15-30분 | ✅ | 연구자, 전체 기능 필요 |

## 🔄 업데이트 방법

### 패키지 업데이트
```bash
uv sync
uv add tensorflow@latest torch@latest
```

### 전체 재설치
```bash
# 기존 환경 삭제 후
install_ai_environment.bat  # 또는 .ps1
```

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

3. **백업 및 복원**
   - `uv export > requirements.txt` - 현재 환경 백업
   - `uv sync` - 백업된 환경 복원

이제 Windows 환경에서도 완벽한 AI Data Scientist 환경을 구축할 수 있습니다! 🚀✨
=======
# agentic_ai
agentic_ai
>>>>>>> a9c279fb1484da2a4e33c488b90affea1906e08f
