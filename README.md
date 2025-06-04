# Cherry AI

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3.23+-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3.25+-orange.svg)
[![Open in - LangGraph Studio](https://img.shields.io/badge/Open_in-LangGraph_Studio-00324d.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4NS4zMzMiIGhlaWdodD0iODUuMzMzIiB2ZXJzaW9uPSIxLjAiIHZpZXdCb3g9IjAgMCA2NCA2NCI+PHBhdGggZD0iTTEzIDcuOGMtNi4zIDMuMS03LjEgNi4zLTYuOCAyNS43LjQgMjQuNi4zIDI0LjUgMjUuOSAyNC41QzU3LjUgNTggNTggNTcuNSA1OCAzMi4zIDU4IDcuMyA1Ni43IDYgMzIgNmMtMTIuOCAwLTE2LjEuMy0xOSAxLjhtMzcuNiAxNi42YzIuOCAyLjggMy40IDQuMiAzLjQgNy42cy0uNiA0LjgtMy40IDcuNkw0Ny4yIDQzSDE2LjhsLTMuNC0zLjRjLTQuOC00LjgtNC44LTEwLjQgMC0xNS4ybDMuNC0zLjRoMzAuNHoiLz48cGF0aCBkPSJNMTguOSAyNS42Yy0xLjEgMS4zLTEgMS43LjQgMi41LjkuNiAxLjcgMS44IDEuNyAyLjcgMCAxIC43IDIuOCAxLjYgNC4xIDEuNCAxLjkgMS40IDIuNS4zIDMuMi0xIC42LS42LjkgMS40LjkgMS41IDAgMi43LS41IDIuNy0xIDAtLjYgMS4xLS44IDIuNi0uNGwyLjYuNy0xLjgtMi45Yy01LjktOS4zLTkuNC0xMi4zLTExLjUtOS44TTM5IDI2YzAgMS4xLS45IDIuNS0yIDMuMi0yLjQgMS41LTIuNiAzLjQtLjUgNC4yLjguMyAyIDEuNyAyLjUgMy4xLjYgMS41IDEuNCAyLjMgMiAyIDEuNS0uOSAxLjItMy41LS40LTMuNS0yLjEgMC0yLjgtMi44LS44LTMuMyAxLjYtLjQgMS42LS41IDAtLjYtMS4xLS4xLTEuNS0uNi0xLjItMS42LjctMS43IDMuMy0yLjEgMy41LS41LjEuNS4yIDEuNi4zIDIuMiAwIC43LjkgMS40IDEuOSAxLjYgMi4xLjQgMi4zLTIuMy4yLTMuMi0uOC0uMy0yLTEuNy0yLjUtMy4xLTEuMS0zLTMtMy4zLTMtLjUiLz48L3N2Zz4=)](https://langgraph-studio.vercel.app/templates/open?githubUrl=https://github.com/langchain-ai/react-agent)

## 프로젝트 개요

> 채팅 인터페이스

`Cherry AI`는 Model Context Protocol(MCP)을 통해 다양한 외부 도구와 데이터 소스에 접근할 수 있는 ReAct 에이전트를 구현한 프로젝트입니다. 주요 기능으로 AI Data Scientist Agent를 제공합니다.


### 주요 기능
 
**동적 방식으로 도구 설정 대시보드**

`http://localhost:2025` 에 접속하여 도구 설정 대시보드를 확인할 수 있습니다.


**도구 추가** 탭에서 [Smithery](https://smithery.io) 에서 사용할 MCP 도구의 JSON 구성을 복사 붙여넣기 하여 도구를 추가할 수 있습니다.

----

**실시간 반영**

도구 설정 대시보드에서 도구를 추가하거나 수정하면 실시간으로 반영됩니다.

**시스템 프롬프트 설정**
도구 설정 대시보드에서 프롬프트를 선택하면 동적으로 반영됩니다.

----

### 주요 기능

* **LangGraph ReAct 에이전트**: LangGraph를 기반으로 하는 ReAct 에이전트
* **실시간 동적 도구 관리**: MCP 도구를 쉽게 추가, 제거, 구성 가능 (Smithery JSON 형식 지원)
* **실시간 동적 시스템 프롬프트 설정**: 시스템 프롬프트를 쉽게 수정 가능 (동적 반영)
* **대화 기록**: 에이전트와의 대화 내용 추적 및 관리
* **localhost 지원**: localhost 로 실행 가능(채팅 인터페이스 연동 가능)

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
'''base
uv run mcp_file_management.py
uv run mcp_private_rag.py
'''


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
