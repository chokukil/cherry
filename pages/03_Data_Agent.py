from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple
from itertools import count
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
import streamlit as st
import glob
import json
import os
from pathlib import Path
from copy import deepcopy
from dotenv import load_dotenv, find_dotenv
import asyncio
import nest_asyncio
import platform
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils import astream_graph, random_uuid
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
import requests
import sys
from contextlib import asynccontextmanager
from mcp.client import stdio as _stdio
import logging
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import ConfigDict
import yaml
from langchain_openai import ChatOpenAI
import pandas as pd
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from urllib.parse import urlsplit, parse_qs
from utils import generate_followups, get_followup_llm
from langchain_experimental.tools import PythonAstREPLTool
from langchain.agents.agent_types import AgentType
import time

from utils import (
    astream_graph, random_uuid, generate_followups, get_followup_llm, 
    PandasAgentStreamParser, AgentCallbacks, pandas_tool_callback, 
    pandas_observation_callback, pandas_result_callback,
    # 추가 import - pandas agent 실시간 처리용
    tool_callback, observation_callback, result_callback,
    AgentStreamParser, ToolChunkHandler, display_message_tree,
    pretty_print_messages, messages_to_history, get_role_from_messages
)

# 🆕 데이터 분석 패키지들 import
import numpy as np
import scipy
import seaborn as sns
import sklearn
from sklearn import datasets, metrics, model_selection, preprocessing, linear_model, ensemble, cluster
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Base directory for app icons
ASSETS_DIR = "assets"
URL_BASE = "http://localhost:2025/Agent?id="

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Log session state initialization
logging.debug('Initializing session state')

if platform.system() == "Windows":
    logging.debug(f"Using proactor: IocpProactor")
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# nest_asyncio 적용: 이미 실행 중인 이벤트 루프 내에서 중첩 호출 허용
nest_asyncio.apply()

# 전역 이벤트 루프 생성 및 재사용 (한번 생성한 후 계속 사용)
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)


st.set_page_config(
    page_title="AI Agent Builder",
    layout="wide",
    initial_sidebar_state="expanded",
)


OUTPUT_TOKEN_INFO = {
    "o4-mini": {"max_tokens": 16000},
    "gpt-4o": {"max_tokens": 16000},
}

# 🆕 데이터 분석 환경 설정 함수
def create_data_analysis_environment(df=None):
    """
    데이터 분석에 필요한 모든 패키지들을 사전에 로드한 환경을 생성합니다.
    plt.show()를 자동으로 Streamlit 호환 버전으로 패치합니다.
    
    Args:
        df: 분석할 DataFrame (선택사항)
    
    Returns:
        dict: 사전 로드된 패키지들과 데이터를 포함한 환경 딕셔너리
    """
    # 🆕 한글 폰트 설정 추가
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import platform
    import warnings
    
    def setup_korean_font():
        """한글 폰트를 설정하는 함수"""
        try:
            # Windows 환경에서 한글 폰트 설정
            if platform.system() == 'Windows':
                # Windows에서 사용 가능한 한글 폰트들 (우선순위 순)
                korean_fonts = ['Malgun Gothic', 'Arial Unicode MS', 'Gulim', 'Dotum', 'Batang']
                
                for font_name in korean_fonts:
                    try:
                        # 폰트가 시스템에 설치되어 있는지 확인
                        available_fonts = [f.name for f in fm.fontManager.ttflist]
                        if font_name in available_fonts:
                            plt.rcParams['font.family'] = font_name
                            plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
                            
                            # 폰트 테스트
                            fig, ax = plt.subplots(figsize=(1, 1))
                            ax.text(0.5, 0.5, '한글테스트', fontsize=10)
                            plt.close(fig)  # 테스트 후 즉시 닫기
                            
                            logging.debug(f"한글 폰트 설정 완료: {font_name}")
                            return True
                    except Exception as e:
                        logging.debug(f"폰트 {font_name} 설정 실패: {e}")
                        continue
                        
            elif platform.system() == 'Darwin':  # macOS
                try:
                    plt.rcParams['font.family'] = 'AppleGothic'
                    plt.rcParams['axes.unicode_minus'] = False
                    logging.debug("한글 폰트 설정 완료: AppleGothic")
                    return True
                except:
                    pass
                    
            elif platform.system() == 'Linux':
                try:
                    # Linux에서 한글 폰트 시도
                    linux_fonts = ['NanumGothic', 'NanumBarunGothic', 'DejaVu Sans']
                    for font_name in linux_fonts:
                        try:
                            plt.rcParams['font.family'] = font_name
                            plt.rcParams['axes.unicode_minus'] = False
                            logging.debug(f"한글 폰트 설정 완료: {font_name}")
                            return True
                        except:
                            continue
                except:
                    pass
            
            # 기본 설정이 실패한 경우
            logging.warning("한글 폰트 설정에 실패했습니다. 기본 폰트를 사용합니다.")
            plt.rcParams['axes.unicode_minus'] = False  # 최소한 마이너스 기호는 보호
            return False
            
        except Exception as e:
            logging.error(f"한글 폰트 설정 중 오류 발생: {e}")
            return False
    
    # 한글 폰트 설정 실행
    setup_korean_font()
    
    # 🆕 폰트 캐시 새로고침 (필요한 경우)
    try:
        # 폰트 캐시가 오래된 경우 새로고침
        fm._rebuild()
    except Exception as e:
        logging.debug(f"폰트 캐시 새로고침 실패 (무시 가능): {e}")
    
    # matplotlib의 원본 show 함수 백업
    original_show = plt.show
    original_clf = plt.clf
    original_cla = plt.cla
    original_close = plt.close
    
    def streamlit_show(*args, **kwargs):
        """
        plt.show()를 Streamlit 환경에서 자동으로 st.pyplot()으로 변환하는 함수
        """
        try:
            # 현재 figure가 있는지 확인
            fig = plt.gcf()
            if fig.get_axes():  # axes가 있으면 실제 플롯이 있다는 의미
                
                # 🆕 현재 진행 중인 메시지에 시각화 추가
                if "current_message_visualizations" not in st.session_state:
                    st.session_state.current_message_visualizations = []
                
                # figure를 base64 이미지로 변환
                import io
                import base64
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='white')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.getvalue()).decode()
                
                # HTML img 태그로 변환 (Streamlit markdown에서 렌더링 가능)
                img_html = f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto; margin: 10px 0;">'
                
                # 현재 메시지의 시각화 목록에 추가
                st.session_state.current_message_visualizations.append(img_html)
                
                # 🆕 시각화 전용 컨테이너에 표시 (실시간)
                if hasattr(st, '_visualization_container') and st._visualization_container is not None:
                    with st._visualization_container:
                        st.pyplot(fig, clear_figure=False)
                else:
                    # 일반적인 경우
                    st.pyplot(fig, clear_figure=False)
                
                # 🆕 새로운 플롯을 위해 새 figure 생성
                plt.figure()
                
            else:
                # 빈 figure인 경우 원래 show 함수 호출
                original_show(*args, **kwargs)
        except Exception as e:
            # 오류 발생 시 원래 show 함수로 fallback
            print(f"Streamlit show error: {e}")
            original_show(*args, **kwargs)
    
    def protected_clf(*args, **kwargs):
        """plt.clf()를 보호하여 의도치 않은 클리어 방지"""
        # 새 figure를 생성하되 기존 것은 건드리지 않음
        plt.figure()
    
    def protected_cla(*args, **kwargs):
        """plt.cla()를 보호하여 의도치 않은 클리어 방지"""
        # 현재 axes만 클리어하되 figure는 유지
        if plt.gcf().get_axes():
            plt.gca().clear()
    
    def protected_close(*args, **kwargs):
        """plt.close()를 보호하여 표시된 figure는 유지"""
        # 인자가 없으면 현재 figure만 닫기
        if not args and not kwargs:
            plt.figure()  # 새 figure 생성
        else:
            original_close(*args, **kwargs)
    
    # matplotlib show 함수를 패치
    plt.show = streamlit_show
    
    # 🆕 matplotlib 클리어 함수들도 패치하여 의도치 않은 figure 삭제 방지
    plt.clf = protected_clf
    plt.cla = protected_cla  
    plt.close = protected_close
    
    # 추가 시각화 헬퍼 함수들
    def reset_show():
        """원본 matplotlib 함수들로 복원"""
        plt.show = original_show
        plt.clf = original_clf
        plt.cla = original_cla
        plt.close = original_close
    
    def force_show():
        """현재 figure를 강제로 Streamlit에 표시"""
        fig = plt.gcf()
        if fig.get_axes():
            st.pyplot(fig, clear_figure=False)
            # 새로운 figure 생성 (기존 것은 유지)
            plt.figure()
    
    # 🆕 한글 폰트 상태 확인 함수
    def check_korean_font():
        """현재 설정된 폰트와 한글 지원 여부를 확인"""
        current_font = plt.rcParams['font.family']
        unicode_minus = plt.rcParams['axes.unicode_minus']
        
        info = f"""
📝 **폰트 설정 정보:**
- 현재 폰트: {current_font}
- 마이너스 기호 보호: {unicode_minus}
- 플랫폼: {platform.system()}

🎨 **한글 테스트**: 가나다라마바사 ← 이 글자들이 정상적으로 보이면 성공!
"""
        return info
    
    # 🆕 데이터 분석 오류 복구용 헬퍼 함수들
    def safe_dataframe_check(obj):
        """DataFrame을 안전하게 체크하는 함수"""
        if obj is None:
            return False
        if hasattr(obj, 'empty'):
            return not obj.empty
        return bool(obj)
    
    def diagnose_data(df=None):
        """데이터 진단 정보를 반환하는 함수"""
        if df is None and 'df' in locals():
            df = locals()['df']
        if df is None and 'data' in locals():
            df = locals()['data']
        if df is None:
            return "진단할 데이터가 없습니다."
        
        try:
            info = f"""
📊 데이터 진단 결과:
- 크기: {df.shape[0]:,} 행 × {df.shape[1]:,} 열  
- 컬럼: {list(df.columns)}
- 데이터 타입: {dict(df.dtypes)}
- 결측값: {dict(df.isnull().sum())}
- 메모리 사용량: {df.memory_usage(deep=True).sum():,} bytes
"""
            return info
        except Exception as e:
            return f"데이터 진단 중 오류: {str(e)}"
    
    def safe_plot():
        """안전한 플롯 생성을 위한 함수"""
        try:
            fig = plt.gcf()
            if hasattr(fig, 'get_axes') and fig.get_axes():
                st.pyplot(fig, clear_figure=False)
                plt.figure()
                return "플롯이 성공적으로 표시되었습니다."
            else:
                return "표시할 플롯이 없습니다."
        except Exception as e:
            return f"플롯 표시 중 오류: {str(e)}"
    
    # 🆕 시각화 관리용 헬퍼 함수들
    def get_current_visualizations():
        """현재 메시지의 시각화 개수 반환"""
        if "current_message_visualizations" in st.session_state:
            return len(st.session_state.current_message_visualizations)
        return 0
    
    def clear_current_visualizations():
        """현재 메시지의 시각화 데이터 제거"""
        if "current_message_visualizations" in st.session_state:
            count = len(st.session_state.current_message_visualizations)
            st.session_state.current_message_visualizations = []
            return f"{count}개의 시각화 데이터가 제거되었습니다."
        return "제거할 시각화 데이터가 없습니다."
    
    def preview_current_visualizations():
        """현재 메시지의 시각화들을 미리보기"""
        if ("current_message_visualizations" in st.session_state and 
            st.session_state.current_message_visualizations):
            st.write(f"**현재 생성된 시각화 {len(st.session_state.current_message_visualizations)}개:**")
            for i, viz_html in enumerate(st.session_state.current_message_visualizations):
                st.markdown(f"시각화 {i+1}:", unsafe_allow_html=False)
                st.markdown(viz_html, unsafe_allow_html=True)
        else:
            st.write("현재 생성된 시각화가 없습니다.")
    
    analysis_env = {
        # 기본 데이터 분석 패키지들
        "pd": pd,
        "pandas": pd,
        "np": np,
        "numpy": np,
        "scipy": scipy,
        "sns": sns,
        "seaborn": sns,
        "plt": plt,
        "matplotlib": plt,
        
        # Streamlit 
        "st": st,
        
        # scikit-learn 관련
        "sklearn": sklearn,
        "datasets": datasets,
        "metrics": metrics,
        "model_selection": model_selection,
        "preprocessing": preprocessing,
        "linear_model": linear_model,
        "ensemble": ensemble,
        "cluster": cluster,
        
        # 기타 유용한 패키지들
        "warnings": warnings,
        "os": os,
        "sys": sys,
        "json": json,
        "time": time,
        
        # 자주 사용하는 함수들을 직접 접근 가능하게
        "train_test_split": model_selection.train_test_split,
        "StandardScaler": preprocessing.StandardScaler,
        "LinearRegression": linear_model.LinearRegression,
        "RandomForestClassifier": ensemble.RandomForestClassifier,
        "KMeans": cluster.KMeans,
        
        # 🆕 시각화 헬퍼 함수들
        "reset_show": reset_show,
        "force_show": force_show,
        "original_show": original_show,
        "original_clf": original_clf,
        "original_cla": original_cla,
        "original_close": original_close,
        
        # 🆕 폰트 관련 헬퍼 함수들
        "setup_korean_font": setup_korean_font,
        "check_korean_font": check_korean_font,
        
        # 🆕 오류 복구용 헬퍼 함수들
        "safe_dataframe_check": safe_dataframe_check,
        "diagnose_data": diagnose_data,
        "safe_plot": safe_plot,
        
        # 🆕 시각화 관리용 헬퍼 함수들
        "get_current_visualizations": get_current_visualizations,
        "clear_current_visualizations": clear_current_visualizations,
        "preview_current_visualizations": preview_current_visualizations,
    }
    
    # DataFrame이 제공된 경우 추가
    if df is not None:
        analysis_env["df"] = df
        analysis_env["data"] = df  # 일반적인 별명도 추가
    
    return analysis_env

# 🆕 데이터 분석용 PythonAstREPLTool 생성 함수
def create_enhanced_python_tool(df=None):
    """
    데이터 분석 패키지들이 사전 로드된 PythonAstREPLTool을 생성합니다.
    plt.show()가 자동으로 Streamlit에서 동작하도록 패치되어 있습니다.
    
    Args:
        df: 분석할 DataFrame (선택사항)
    
    Returns:
        PythonAstREPLTool: 향상된 Python REPL 도구
    """
    analysis_env = create_data_analysis_environment(df)
    
    # 사용자 친화적인 설명과 예제 추가
    description = """
    🤖 **지능형 데이터 분석 환경**에 오신 것을 환영합니다!
    
    📊 **사전 로드된 패키지들:**
    - 데이터 처리: pandas (pd), numpy (np)
    - 시각화: matplotlib (plt), seaborn (sns), streamlit (st)  
    - 머신러닝: scikit-learn (sklearn)
    - 과학계산: scipy
    
    🚀 **특별 기능들:**
    ✅ CSV 업로드 시 **수동 EDA(탐색적 데이터 분석)** 버튼 제공
    ✅ **데이터 특성 자동 분석** 및 **분석 방향 추천**  
    ✅ plt.show() 자동 Streamlit 변환 (시각화 영구 보존)
    ✅ **맞춤형 후속 질문** 자동 생성
    ✅ plt.clf(), plt.cla(), plt.close() 등 클리어 함수들로부터 보호
    ✅ 도구 호출 정보가 접혀도 시각화는 그대로 유지!
    
    📈 **자동 분석 항목:**
    - 데이터 크기, 타입, 결측값 현황
    - 수치형/범주형/날짜형 컬럼 분류  
    - 기본 통계 요약 및 분포 시각화
    - 분석 우선순위 및 방향 제안
    
    🎯 **시작 방법:**
    1. CSV 파일 업로드 → Agent 생성
    2. 사이드바의 '🚀 자동 데이터 분석 시작' 버튼 클릭
    3. 제안된 분석 중 원하는 것 선택
    4. 대화형으로 심화 분석 진행
    
    💬 **사용 예시:**
    - "결측값을 처리해줘"
    - "상관관계를 시각화해줘" 
    - "이상치를 찾아줘"
    - "클러스터링 분석을 해줘"
    
    DataFrame은 'df' 또는 'data' 변수로 접근할 수 있습니다.
    무엇을 분석하고 싶으신지 말씀해 주세요! 🤖✨
    """
    
    return PythonAstREPLTool(
        locals=analysis_env,
        description=description,
        name="enhanced_python_repl",
        handle_tool_error=True
    )

# 🆕 자동 데이터 분석 및 인사말 생성 함수
def auto_analyze_and_greet(df):
    """
    데이터 로드 시 자동으로 기본 분석을 수행하고 인사말과 가이드를 생성합니다.
    """
    try:
        # 데이터 기본 정보 수집
        shape = df.shape
        columns = df.columns.tolist()
        dtypes = df.dtypes.value_counts().to_dict()
        missing_values = df.isnull().sum().sum()
        missing_cols = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
        
        # 수치형/범주형 컬럼 분류
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # 메모리 사용량
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        # 인사말 및 분석 결과 생성
        greeting_content = f"""🎉 **데이터 분석 환경에 오신 것을 환영합니다!**

📊 **로드된 데이터 개요:**
- **데이터 크기**: {shape[0]:,} 행 × {shape[1]:,} 열
- **메모리 사용량**: {memory_usage:.2f} MB
- **결측값**: {missing_values:,} 개 ({missing_values/df.size*100:.1f}%)

📋 **컬럼 구성:**
- **수치형 컬럼** ({len(numeric_cols)}개): {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}
- **범주형 컬럼** ({len(categorical_cols)}개): {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}
{'- **날짜형 컬럼** (' + str(len(datetime_cols)) + '개): ' + ', '.join(datetime_cols[:3]) + ('...' if len(datetime_cols) > 3 else '') if datetime_cols else ''}

🔍 **데이터 미리보기:**"""

        # 데이터 미리보기를 텍스트로 포함
        preview_text = df.head(3).to_string()
        greeting_content += f"\n```\n{preview_text}\n```\n"
        
        # 분석 제안 생성
        suggestions = []
        
        if missing_values > 0:
            suggestions.append(f"📍 **결측값 처리**: {len(missing_cols)}개 컬럼에 결측값이 있습니다")
            
        if len(numeric_cols) >= 2:
            suggestions.append("📈 **상관관계 분석**: 수치형 변수들 간의 상관관계를 확인해보세요")
            
        if len(categorical_cols) > 0:
            suggestions.append("📊 **범주형 데이터 분포**: 카테고리별 빈도와 분포를 살펴보세요")
            
        if len(numeric_cols) > 0:
            suggestions.append("📉 **기초 통계**: 수치형 데이터의 분포와 이상치를 확인해보세요")
            
        if shape[0] > 1000:
            suggestions.append("🎯 **샘플링**: 큰 데이터셋이므로 샘플링을 고려해보세요")
            
        # 구체적인 분석 명령어 제안
        greeting_content += "\n💡 **추천 분석 단계:**\n"
        for i, suggestion in enumerate(suggestions[:4], 1):
            greeting_content += f"{i}. {suggestion}\n"
            
        greeting_content += """
🚀 **빠른 시작 명령어:**
- `df.describe()` - 기초 통계 요약
- `df.info()` - 데이터 타입 및 결측값 정보  
- `df.hist(figsize=(12, 8)); plt.show()` - 전체 변수 히스토그램
- `sns.heatmap(df.corr(), annot=True); plt.show()` - 상관관계 히트맵
- `df.isnull().sum()` - 결측값 확인

무엇을 분석하고 싶으신지 말씀해 주세요! 🤖✨"""

        # 시각화 생성을 위한 기본 플롯
        visualizations = []
        try:
            # 간단한 데이터 개요 시각화 생성
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 1. 데이터 타입 분포 파이차트
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 컬럼 타입 분포
            type_counts = {'수치형': len(numeric_cols), '범주형': len(categorical_cols), '날짜형': len(datetime_cols)}
            type_counts = {k: v for k, v in type_counts.items() if v > 0}
            
            ax1.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%', startangle=90)
            ax1.set_title('컬럼 타입 분포')
            
            # 결측값 현황
            if missing_values > 0 and len(missing_cols) <= 10:
                missing_data = pd.Series(missing_cols)
                missing_data.plot(kind='bar', ax=ax2, color='coral')
                ax2.set_title('컬럼별 결측값 개수')
                ax2.set_ylabel('결측값 개수')
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            else:
                ax2.text(0.5, 0.5, f'전체 결측값: {missing_values:,}개\n({missing_values/df.size*100:.1f}%)', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('결측값 현황')
                ax2.axis('off')
            
            plt.tight_layout()
            
            # figure를 base64로 변환하여 저장
            import io
            import base64
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='white')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode()
            img_html = f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto; margin: 10px 0;">'
            
            visualizations.append(img_html)
            plt.close(fig)  # 메모리 정리
            
        except Exception as viz_error:
            logging.warning(f"초기 시각화 생성 실패: {viz_error}")
        
        # 분석 결과를 세션 상태에 저장 (history에는 추가하지 않음)
        analysis_result = {
            "content": greeting_content,
            "visualizations": visualizations,
            "followups": [
                "데이터의 기초 통계를 보여줘",
                "결측값이 있는 컬럼들을 확인해줘", 
                "수치형 변수들의 상관관계를 시각화해줘",
                "전체 데이터의 히스토그램을 그려줘"
            ]
        }
        
        # 🆕 분석 결과를 별도 저장 (초기화 완료 후 사용)
        st.session_state.auto_analysis_result = analysis_result
        
        return True
        
    except Exception as e:
        logging.error(f"자동 데이터 분석 중 오류: {e}")
        # 간단한 인사말만 저장
        simple_greeting = f"""🎉 **데이터 분석 환경에 오신 것을 환영합니다!**

📊 **데이터가 성공적으로 로드되었습니다**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열

데이터를 분석하고 싶은 내용을 말씀해 주세요! 🤖✨"""
        
        st.session_state.auto_analysis_result = {
            "content": simple_greeting,
            "visualizations": [],
            "followups": ["데이터의 기본 정보를 보여줘", "첫 5행을 보여줘", "데이터 요약 통계를 보여줘"]
        }
        
        return False

# Log function entry and exit
logging.debug('Entering function: initialize_session')
async def initialize_session(mcp_config=None):
    logging.debug('Initializing MCP session')
    with st.spinner("🔄 MCP 서버에 연결 중..."):
        await cleanup_mcp_client()
        logging.debug('MCP client cleaned up')

        # mcp_config이 None이거나 tool_config가 없는 경우 MCP 연결을 건너뜁니다.
        if mcp_config is None and (
            "tool_config" not in st.session_state or st.session_state.tool_config is None
        ):
            st.warning("⚠️ MCP 서버 연결을 건너뜁니다. 사이드바에서 MCP Tool을 선택해주세요.")
            st.session_state.tool_count = 0
            st.session_state.mcp_client = None
            st.session_state.session_initialized = True
            logging.debug('No tool configuration found, skipping MCP connection.')
            return True

        # mcp_config이 None이면 사이드바에서 로드된 tool_config 사용
        if mcp_config is None:
            mcp_config = st.session_state.tool_config

        # mcpServers 키가 있으면 해제
        connections = mcp_config.get("mcpServers", mcp_config)
        
        # Store connections for debugging
        st.session_state.last_mcp_connections = connections
        logging.debug(f"MCP connections configuration: {json.dumps(connections, indent=2)}")
        
        # MCP 서버 설정이 비어 있으면 건너뜁니다.
        if not connections:
            st.warning("⚠️ MCP 서버 설정이 비어 있습니다. MCP 연결을 건너뜁니다.")
            st.session_state.tool_count = 0
            st.session_state.mcp_client = None
            st.session_state.session_initialized = True
            logging.debug('MCP server configuration is empty, skipping connection.')
            return True

        # Initialize MCP client and connect to servers
        try:
            logging.debug("Creating MultiServerMCPClient with connections")
            client = MultiServerMCPClient(connections)
            
            logging.debug("Entering MCP client context")
            await client.__aenter__()
            logging.debug('MCP servers connected via context manager.')
            
            try:
                # Get and log available tools
                logging.debug("Retrieving tools from MCP client")
                tools = client.get_tools()
                tool_count = len(tools)
                st.session_state.tool_count = tool_count
                
                # Log individual tool details
                logging.debug(f"Retrieved {tool_count} tools from MCP client")
                for i, tool in enumerate(tools):
                    tool_name = getattr(tool, 'name', f"Tool_{i}")
                    logging.debug(f"Tool {i}: {tool_name}")
                    if hasattr(tool, 'args'):
                        logging.debug(f"Tool {i} args: {tool.args}")
                    if hasattr(tool, 'description'):
                        logging.debug(f"Tool {i} description: {tool.description}")
                
                st.session_state.mcp_client = client
            except Exception as e:
                logging.error(f"Error retrieving tools: {str(e)}")
                import traceback
                logging.error(f"Tool retrieval error details:\n{traceback.format_exc()}")
                st.error(f"MCP 도구를 가져오는 중 오류가 발생했습니다: {str(e)}")
                # Continue with empty tools list
                tools = []
                tool_count = 0
                st.session_state.tool_count = 0
            
            # Create agent based on whether DataFrame is available
            if tool_count > 0 or st.session_state.dataframe is not None:  # 🆕 DataFrame이 있으면 도구가 없어도 에이전트 생성
                # Replace HTTPChatModel usage with ChatOpenAI
                load_dotenv()
                # Construct OpenAI API base URL (always host + /v1)
                raw_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com")
                # Remove any path, keep only scheme and netloc
                parsed = urlsplit(raw_base)
                openai_api_base = f"{parsed.scheme}://{parsed.netloc}/v1"
                openai_api_key = os.getenv("OPENAI_API_KEY", "")
                logging.debug(f"Creating ChatOpenAI with base_url: {openai_api_base}")
                model = ChatOpenAI(
                    model=st.session_state.selected_model,
                    temperature=st.session_state.temperature,
                    max_tokens=OUTPUT_TOKEN_INFO[st.session_state.selected_model]["max_tokens"],
                    api_key=openai_api_key,
                    base_url=openai_api_base,
                    streaming=True,  # Enable streaming
                )
                try:
                    # --- 에이전트 생성 ---
                    if st.session_state.dataframe is not None:          # 🆕 DataFrame이 있으면
                        df = st.session_state.dataframe
                        
                        # 🆕 데이터 분석용 도구 생성
                        enhanced_python_tool = create_enhanced_python_tool(df)
                        
                        # Create pandas agent with enhanced tools integration
                        # First create the base pandas agent
                        pandas_agent = create_pandas_dataframe_agent(
                            model,
                            df,
                            verbose=True,
                            agent_type=AgentType.OPENAI_FUNCTIONS,
                            allow_dangerous_code=True,
                            prefix=st.session_state.selected_prompt_text,
                            handle_parsing_errors=True,
                            max_iterations=10,
                            early_stopping_method="generate"
                        )
                        
                        # 🆕 향상된 도구들을 pandas 에이전트에 추가
                        enhanced_tools = [enhanced_python_tool]
                        
                        # Add MCP tools if available
                        if tools:
                            enhanced_tools.extend(tools)
                            logging.debug(f"Added {len(tools)} MCP tools to pandas agent")
                        
                        # Get existing tools from pandas agent and combine
                        existing_tools = pandas_agent.tools if hasattr(pandas_agent, 'tools') else []
                        combined_tools = existing_tools + enhanced_tools
                        
                        # Update the agent with combined tools
                        if hasattr(pandas_agent, 'tools'):
                            pandas_agent.tools = combined_tools
                        elif hasattr(pandas_agent, 'agent') and hasattr(pandas_agent.agent, 'tools'):
                            pandas_agent.agent.tools = combined_tools
                        
                        logging.debug(f"Enhanced pandas agent created with {len(combined_tools)} total tools")
                        
                        # Ensure the agent supports streaming
                        if hasattr(pandas_agent, 'llm'):
                            pandas_agent.llm.streaming = True
                        elif hasattr(pandas_agent, 'agent') and hasattr(pandas_agent.agent, 'llm_chain') and hasattr(pandas_agent.agent.llm_chain, 'llm'):
                            pandas_agent.agent.llm_chain.llm.streaming = True
                            
                        st.session_state.agent = pandas_agent
                        st.session_state.agent_type = "pandas"
                        logging.debug('Enhanced pandas agent with pre-loaded packages created successfully')
                        
                        # 🆕 사용 가능한 패키지 정보를 사용자에게 표시
                        st.sidebar.success("✅ 지능형 데이터 분석 환경 준비 완료!")
                        
                        with st.sidebar.expander("📦 사전 로드된 패키지", expanded=False):
                            st.write("""
                            **데이터 처리:**
                            - pandas (pd), numpy (np)
                            
                            **시각화:**
                            - matplotlib (plt), seaborn (sns)
                            - ✨ plt.show() 자동 Streamlit 변환 (영구 보존)
                            
                            **머신러닝:**
                            - scikit-learn (sklearn)
                            - datasets, metrics, model_selection
                            - preprocessing, linear_model, ensemble, cluster
                            
                            **과학계산:**
                            - scipy
                            
                            **추천 시작 명령어:**
                            - `df.describe()` - 기초 통계 요약
                            - `df.hist(); plt.show()` - 히스토그램
                            - `sns.heatmap(df.corr()); plt.show()` - 상관관계
                            """)
                        
                    else:                                               # 없으면 기존 ReAct 유지
                        # 🆕 일반 에이전트에도 향상된 Python 도구 추가
                        enhanced_tools = tools.copy() if tools else []
                        enhanced_python_tool = create_enhanced_python_tool()
                        enhanced_tools.append(enhanced_python_tool)
                        
                        agent = create_react_agent(
                            model,
                            enhanced_tools,
                            prompt=st.session_state.selected_prompt_text,
                            checkpointer=MemorySaver(),
                        )
                        st.session_state.agent = agent
                        st.session_state.agent_type = "langgraph"
                        logging.debug('Enhanced LangGraph ReAct agent created successfully')
                        
                except Exception as e:
                    logging.error(f"Error creating agent: {str(e)}")
                    import traceback
                    logging.error(f"Agent creation error details:\n{traceback.format_exc()}")
                    st.error(f"에이전트 생성 중 오류가 발생했습니다: {str(e)}")
                    st.session_state.agent = None
                    st.session_state.agent_type = None
            else:
                st.session_state.agent = None
                st.session_state.agent_type = None
                logging.warning('No tools available and no DataFrame loaded, agent not created.')
            
            st.session_state.session_initialized = True
            return True
            
        except Exception as e:
            import traceback
            error_msg = f"Error initializing MCP client: {str(e)}"
            logging.error(f"{error_msg}\n{traceback.format_exc()}")
            st.error(error_msg)
            st.session_state.session_initialized = False
            return False

# Log function entry and exit
logging.debug('Entering function: cleanup_mcp_client')
async def cleanup_mcp_client():
    """
    기존 MCP 클라이언트를 안전하게 종료합니다.

    기존 클라이언트가 있는 경우 정상적으로 리소스를 해제합니다.
    """
    if "mcp_client" in st.session_state and st.session_state.mcp_client is not None:
        try:
            await st.session_state.mcp_client.__aexit__(None, None, None)
            st.session_state.mcp_client = None
        except Exception as e:
            import traceback
            logging.error(f"MCP client cleanup error: {str(e)}\n{traceback.format_exc()}")
    logging.debug('Exiting function: cleanup_mcp_client')


def print_message():
    """
    채팅 기록을 화면에 출력합니다.

    사용자와 어시스턴트의 메시지를 구분하여 화면에 표시하고,
    도구 호출 정보는 어시스턴트 메시지 컨테이너 내에 표시합니다.
    """
    i = 0
    while i < len(st.session_state.history):
        message = st.session_state.history[i]

        if message["role"] == "user":
            st.chat_message("user", avatar="🧑🏻").write(message["content"])
            i += 1
        elif message["role"] == "assistant":
            # 어시스턴트 메시지 컨테이너 생성
            with st.chat_message("assistant", avatar="🤖"):
                # 🆕 메시지 내용을 HTML로 렌더링 (시각화 포함)
                content = message["content"]
                
                # 시각화가 포함된 경우 HTML로 렌더링
                if "visualizations" in message and message["visualizations"]:
                    # 텍스트 내용 먼저 표시
                    if content and content.strip():
                        st.write(content)
                    
                    # 시각화들을 HTML로 표시
                    for viz_html in message["visualizations"]:
                        st.markdown(viz_html, unsafe_allow_html=True)
                else:
                    # 일반 텍스트만 있는 경우
                    st.write(content)

                # --- Followup 버튼 렌더링 ---
                followups = message.get("followups")
                if followups:
                    st.markdown("<div style='margin-top: 0.5em; margin-bottom: 0.5em; color: #888;'>후속 질문 제안:</div>", unsafe_allow_html=True)
                    btn_cols = st.columns(len(followups))
                    for idx, followup in enumerate(followups):
                        if btn_cols[idx].button(followup, key=f"followup_{i}_{idx}"):
                            st.session_state["user_query"] = followup
                            st.rerun()

                # 다음 메시지가 도구 호출 정보인지 확인
                if (
                    i + 1 < len(st.session_state.history)
                    and st.session_state.history[i + 1]["role"] == "assistant_tool"
                ):
                    # 도구 호출 정보를 동일한 컨테이너 내에 expander로 표시
                    with st.expander("🔧 도구 호출 정보", expanded=False):
                        st.write(st.session_state.history[i + 1]["content"])
                    i += 2  # 두 메시지를 함께 처리했으므로 2 증가
                else:
                    i += 1  # 일반 메시지만 처리했으므로 1 증가
        else:
            # assistant_tool 메시지는 위에서 처리되므로 건너뜀
            i += 1


def get_streaming_callback(text_placeholder, tool_placeholder) -> Tuple:

    text_buf: List[str] = []
    tool_buf:  List[str] = []

    live: Dict[str, Dict[str, Any]] = {}     # id → info
    seq_counter = count(1)                   # 시각적 순서 번호
    PENDING_ID = "__pending"                 # 임시 입력용 단일 키

    # ─── 헬퍼 --------------------------------------------------
    def flush_txt():  text_placeholder.write("".join(text_buf))
    def flush_tool(): tool_placeholder.write("".join(tool_buf))

    def _safe_pretty(raw):
        if raw is None:
            return "{}"
        if isinstance(raw, (dict, list)):
            return json.dumps(raw, ensure_ascii=False, indent=2)
        if isinstance(raw, str):
            if raw.strip() == "":
                return raw
            try:
                return json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
            except Exception:
                return raw
        return str(raw)

    def _reserve_box():
        tool_buf.append("")
        return len(tool_buf) - 1

    # ─── live 관리 -------------------------------------------
    def _ensure_live(tid: str, name: str | None, pending=False):
        if tid not in live:
            live[tid] = {
                "name": name or "Unnamed Tool",
                "args": "",
                "idx": _reserve_box(),
                "seq": next(seq_counter),
                "pending": pending,
            }
        elif name and live[tid]["name"] == "Unnamed Tool":
            live[tid]["name"] = name
        return live[tid]

    def _render_input(info: Dict[str, Any], tid_render: str):
        tool_buf[info["idx"]] = (
            f"\n**📝 도구 입력 [Tool: {info['name']} / ID: {tid_render}]**\n"
            f"```json\n{_safe_pretty(info['args'])}\n```\n"
        )
        flush_tool()

    # ─── accumulate ------------------------------------------
    def _accumulate(tid_raw, name, piece):
        if piece in (None, "") or (isinstance(piece, str) and not piece.strip()):
            return

        has_real_id = bool(tid_raw) and str(tid_raw).lower() != "none"

        # ① 실제 ID가 아직 없을 때 → __pending 이용
        if not has_real_id:
            info = _ensure_live(PENDING_ID, name, pending=True)
            info["args"] += str(piece) if not isinstance(piece, (dict, list)) else piece
            _render_input(info, PENDING_ID)
            return

        # ② 실제 ID가 도착했을 때
        real_id = tid_raw
        info = live.get(real_id)

        #   ②-a: pending 박스가 있으면 승격
        if info is None and PENDING_ID in live:
            info = live.pop(PENDING_ID)
            info["pending"] = False
            live[real_id] = info
        #   ②-b: 처음 보는 ID라면 새로 생성
        if info is None:
            info = _ensure_live(real_id, name)

        # 이름 보강
        if name and info["name"] == "Unnamed Tool":
            info["name"] = name

        # args 이어붙이기
        if isinstance(piece, (dict, list)):
            info["args"] = piece
        else:
            info["args"] += str(piece)

        _render_input(info, real_id)

    # ─── finalize --------------------------------------------
    def _finalize(tid_raw, tname, raw_res):
        tid = tid_raw or ""

        # pending 박스가 결과와 함께 확정될 수도 있음
        if tid not in live and PENDING_ID in live:
            live[tid] = live.pop(PENDING_ID)
            live[tid]["pending"] = False

        info = live.pop(tid, None)
        if info:
            if tname and info["name"] == "Unnamed Tool":
                info["name"] = tname
            _render_input(info, tid)

        tool_buf.append(
            f"\n**✅ 도구 호출 결과 [Tool: {tname or 'Unnamed Tool'} / ID: {tid or 'unknown'}]**\n"
            f"```json\n{_safe_pretty(raw_res)}\n```\n"
        )
        flush_tool()

    # ─── pandas 전용 ----------------------------------------
    from utils import AgentStreamParser, AgentCallbacks
    parser = AgentStreamParser(
        AgentCallbacks(
            lambda t: _pd_tool(t),
            lambda o: _pd_obs(o),
            lambda r: _pd_res(r)
        )
    )

    def _pd_tool(t):
        pretty = json.dumps(t["tool_input"], ensure_ascii=False, indent=2)
        pid = f"pandas_{next(seq_counter)}"
        tool_buf.append(
            f"\n**🔧 도구 호출 [Tool: {t['tool']} / ID: {pid}]**\n```json\n{pretty}\n```\n"
        )
        flush_tool()

    def _pd_obs(o):
        obs = o["observation"]
        if hasattr(obs, "empty") and obs.empty:
            return
        s = str(obs)[:1000] + ("…" if len(str(obs)) > 1000 else "")
        tool_buf.append(f"\n**📊 실행 결과:**\n```\n{s}\n```\n")
        flush_tool()

    def _pd_res(r):
        if isinstance(r, str) and r.strip():
            text_buf.append(r)
            flush_txt()

    # ─── 메인 콜백 ------------------------------------------
    def callback(msg: Dict[str, Any]):
        node, content = msg.get("node", ""), msg.get("content")

        # 1) pandas agent
        if node == "pandas_agent" and isinstance(content, dict):
            parser.process_agent_steps(content)
            return

        # 2) 최종 AIMessage
        if isinstance(content, AIMessage):
            if isinstance(content.content, str):
                text_buf.append(content.content)
                flush_txt()
            for call in content.additional_kwargs.get("tool_calls", []):
                _accumulate(call.get("id"), call["function"]["name"], call["function"]["arguments"])
            return

        # 3) 스트리밍 AIMessageChunk
        if isinstance(content, AIMessageChunk):
            if isinstance(content.content, str):
                text_buf.append(content.content)
                flush_txt()
            for ch in getattr(content, "tool_call_chunks", []):
                _accumulate(ch.get("id"), ch.get("name"), ch.get("args", ""))
            return

        # 4) ToolMessage
        if isinstance(content, ToolMessage):
            _finalize(content.tool_call_id, content.name, content.content)
            return

    # ─────────────────────────────────────────────────────────
    return callback, text_buf, tool_buf


# Handle tool execution
async def execute_tool(tool_call, tools):
    """Execute a tool and return its response"""
    tool_id = tool_call.get("id")
    tool_name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})
    
    logging.debug(f"Executing tool: {tool_name} (ID: {tool_id})")
    logging.debug(f"Arguments: {arguments}")
    
    # Find the matching tool
    matching_tool = None
    for tool in tools:
        if getattr(tool, "name", "") == tool_name:
            matching_tool = tool
            break
    
    if not matching_tool:
        error_msg = f"Tool {tool_name} not found"
        logging.error(error_msg)
        return {
            "tool_call_id": tool_id,
            "name": tool_name,
            "content": f"Error: {error_msg}"
        }
    
    try:
        # Execute the tool with provided arguments
        result = await matching_tool.ainvoke(arguments)
        logging.debug(f"Tool execution result: {str(result)[:200]}...")
        
        # Create response
        return {
            "tool_call_id": tool_id,
            "name": tool_name,
            "content": str(result)
        }
    except Exception as e:
        import traceback
        error_msg = f"Error executing tool {tool_name}: {str(e)}"
        error_trace = traceback.format_exc()
        logging.error(f"{error_msg}\n{error_trace}")
        return {
            "tool_call_id": tool_id,
            "name": tool_name,
            "content": f"Error: {error_msg}"
        }


# Log function entry and exit
logging.debug('Entering function: process_query')
async def process_query(query, text_placeholder, tool_placeholder, timeout_seconds=60):
    """
    사용자 질문을 처리하고 응답을 생성합니다.

    이 함수는 사용자의 질문을 에이전트에 전달하고, 응답을 실시간으로 스트리밍하여 표시합니다.
    지정된 시간 내에 응답이 완료되지 않으면 타임아웃 오류를 반환합니다.

    매개변수:
        query: 사용자가 입력한 질문 텍스트
        text_placeholder: 텍스트 응답을 표시할 Streamlit 컴포넌트
        tool_placeholder: 도구 호출 정보를 표시할 Streamlit 컴포넌트
        timeout_seconds: 응답 생성 제한 시간(초)

    반환값:
        response: 에이전트의 응답 객체
        final_text: 최종 텍스트 응답
        final_tool: 최종 도구 호출 정보
    """
    try:
        if st.session_state.agent:
            logging.debug(f"Processing query: {query}")
            streaming_callback, accumulated_text_obj, accumulated_tool_obj = (
                get_streaming_callback(text_placeholder, tool_placeholder)
            )
            
            # Reset tool tracking for new query
            st.session_state.pending_tool_calls = []
            st.session_state.tool_responses = {}
            
            # 🆕 새 메시지를 위해 시각화 데이터 초기화
            st.session_state.current_message_visualizations = []
            
            # Show initial progress
            text_placeholder.markdown("🤔 질문을 분석하고 있습니다...")
            time.sleep(0.5)
            
            try:
                logging.debug(f"Agent type: {type(st.session_state.agent)}")
                
                # Update progress
                text_placeholder.markdown("🔍 답변을 생성하고 있습니다...")
                
                # Check if this is a pandas agent or regular LangGraph agent
                agent_type = st.session_state.get("agent_type", "unknown")
                
                if agent_type == "pandas":
                    # Handle pandas agent streaming
                    logging.debug("Processing pandas agent query")
                    
                    # Clear the progress message
                    text_placeholder.markdown("")
                    
                    try:
                        # Use agent.stream() for pandas agent
                        response_stream = st.session_state.agent.stream({"input": query})
                        final_output = ""
                        error_occurred = False
                        error_message = ""
                        
                        # Process each step in the stream
                        for step in response_stream:
                            logging.debug(f"Pandas agent step: {type(step)} - {list(step.keys()) if isinstance(step, dict) else step}")
                            
                            try:
                                # Process the step using our callback
                                streaming_callback({"node": "pandas_agent", "content": step})
                                
                                # Extract final output if available
                                if isinstance(step, dict) and "output" in step:
                                    final_output = step["output"]
                                    
                            except Exception as step_error:
                                # 🆕 개별 step 처리 중 오류 발생 시 로깅하고 계속 진행
                                error_occurred = True
                                error_message = str(step_error)
                                logging.warning(f"Error processing pandas agent step: {error_message}")
                                
                                # 🆕 오류 정보를 도구 placeholder에 표시
                                error_entry = f"\n**⚠️ 처리 중 오류 발생:**\n```\n{error_message}\n```\n"
                                tool_placeholder.markdown(error_entry)
                                
                                # 스트림 처리를 계속하되 오류를 기록
                                continue
                        
                        # 🆕 오류가 발생했지만 부분적으로 결과가 있는 경우
                        if error_occurred and final_output:
                            final_output += f"\n\n⚠️ 처리 중 일부 오류가 발생했지만 결과를 생성했습니다: {error_message}"
                            
                        # 🆕 오류가 발생하고 결과가 없는 경우 재시도 또는 대안 제시
                        elif error_occurred and not final_output:
                            final_output = f"❌ 처리 중 오류가 발생했습니다: {error_message}\n\n💡 다음을 시도해보세요:\n- 더 구체적인 질문으로 다시 시도\n- 데이터 형태나 컬럼명 확인\n- 단계별로 나누어 질문"
                            
                        # Ensure final text is displayed
                        if final_output and not accumulated_text_obj:
                            accumulated_text_obj.append(final_output)
                            text_placeholder.markdown("".join(accumulated_text_obj))
                        
                        response = {"output": final_output}
                        
                    except Exception as e:
                        import traceback
                        error_msg = f"Pandas agent 처리 중 오류 발생: {str(e)}"
                        error_trace = traceback.format_exc()
                        logging.error(f"{error_msg}\n{error_trace}")
                        
                        # 🆕 오류 발생 시 사용자에게 도움이 되는 정보 제공
                        user_friendly_error = f"""❌ 데이터 분석 중 오류가 발생했습니다.

**오류 내용:** {str(e)}

💡 **해결 방법:**
1. **데이터 확인**: `df.head()`, `df.info()`, `df.describe()` 로 데이터 상태 확인
2. **컬럼명 확인**: `df.columns.tolist()` 로 정확한 컬럼명 확인  
3. **단계별 접근**: 복잡한 분석을 단계별로 나누어 수행
4. **구체적 질문**: "특정 컬럼의 평균값은?" 같이 구체적으로 질문

**재시도 예시:**
- "데이터의 기본 정보를 보여줘"
- "첫 5행을 보여줘" 
- "컬럼 이름을 알려줘"
"""
                        
                        # 🆕 자동으로 기본 데이터 정보 확인 시도
                        if st.session_state.dataframe is not None:
                            try:
                                df = st.session_state.dataframe
                                auto_info = f"""

🔍 **자동 데이터 진단:**
- **데이터 크기**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- **컬럼명**: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}
- **데이터 타입**: {df.dtypes.value_counts().to_dict()}
- **결측값**: {df.isnull().sum().sum():,} 개
"""
                                user_friendly_error += auto_info
                                
                                # 시각화가 포함된 오류인 경우 추가 정보
                                if "DataFrame" in str(e) and ("empty" in str(e) or "truth" in str(e)):
                                    user_friendly_error += "\n📊 **시각화 관련 팁**: DataFrame이 비어있거나 조건 확인 시 `.empty`, `.any()`, `.all()` 사용"
                                    
                            except Exception as info_error:
                                logging.warning(f"Failed to get automatic data info: {info_error}")
                        
                        accumulated_text_obj.append(user_friendly_error)
                        text_placeholder.markdown("".join(accumulated_text_obj))
                        
                        response = {"output": user_friendly_error, "error": error_msg}
                        
                        
                else:
                    # Handle regular LangGraph agent
                    logging.debug("Processing LangGraph agent query")
                    config = RunnableConfig(
                        recursion_limit=st.session_state.recursion_limit,
                        thread_id=st.session_state.thread_id,
                        configurable={
                            "callbacks": [
                                lambda x: logging.debug(f"RunnableConfig callback: {str(x)[:100]}...")
                            ]
                        }
                    )
                    logging.debug(f"Starting agent execution with timeout: {timeout_seconds}s")
                    
                    # ReAct agent expects a string for HumanMessage
                    agent_task = astream_graph(
                        st.session_state.agent,
                        {"messages": [HumanMessage(content=query)]},
                        callback=streaming_callback,
                        config=config,
                    )
                    
                    response = None
                    try:
                        start_time = asyncio.get_event_loop().time()
                        remaining_time = timeout_seconds
                        response = await asyncio.wait_for(
                            agent_task,
                            timeout=remaining_time
                        )
                        logging.debug("Initial agent response received")
                        
                        # Handle any pending tool calls (for non-pandas agents)
                        if "pending_tool_calls" in st.session_state and st.session_state.pending_tool_calls:
                            while st.session_state.pending_tool_calls and remaining_time > 0:
                                logging.debug(f"Processing pending tool calls: {len(st.session_state.pending_tool_calls)}")
                                tool_call = st.session_state.pending_tool_calls[0]
                                logging.debug(f"Processing tool call: {tool_call}")
                                current_time = asyncio.get_event_loop().time()
                                elapsed = current_time - start_time
                                remaining_time = timeout_seconds - elapsed
                                if remaining_time <= 0:
                                    logging.warning("Tool execution timeout")
                                    break
                                tool_result = await asyncio.wait_for(
                                    execute_tool(tool_call, st.session_state.mcp_client.get_tools()),
                                    timeout=remaining_time
                                )
                                tool_message = ToolMessage(
                                    content=tool_result["content"],
                                    name=tool_result["name"],
                                    tool_call_id=tool_result["tool_call_id"]
                                )
                                with tool_placeholder.expander("🔧 도구 실행 결과", expanded=True):
                                    st.write(f"**도구**: {tool_result['name']}\n\n**결과**:\n```\n{tool_result['content'][:1000]}...\n```")
                                current_time = asyncio.get_event_loop().time()
                                elapsed = current_time - start_time
                                remaining_time = timeout_seconds - elapsed
                                if remaining_time <= 0:
                                    logging.warning("Agent continuation timeout")
                                    break
                                agent_continue_task = astream_graph(
                                    st.session_state.agent,
                                    {"messages": [tool_message]},
                                    callback=streaming_callback,
                                    config=config,
                                )
                                response = await asyncio.wait_for(
                                    agent_continue_task,
                                    timeout=remaining_time
                                )
                                st.session_state.pending_tool_calls = st.session_state.pending_tool_calls[1:]
                                current_time = asyncio.get_event_loop().time()
                                elapsed = current_time - start_time
                                remaining_time = timeout_seconds - elapsed
                                if not st.session_state.pending_tool_calls:
                                    logging.debug("No more pending tool calls")
                                    break
                                    
                    except asyncio.TimeoutError:
                        error_msg = f"⏱️ 요청 시간이 {timeout_seconds}초를 초과했습니다. 나중에 다시 시도해 주세요."
                        logging.error(f"Query timed out after {timeout_seconds} seconds")
                        return {"error": error_msg}, error_msg, ""
                        
                logging.debug("Query completed successfully")
                if hasattr(response, 'get'):
                    resp_content = response.get('content', 'No content')
                    logging.debug(f"Response content: {str(resp_content)[:100]}...")
                else:
                    logging.debug(f"Response type: {type(response)}")
                    
            except asyncio.TimeoutError:
                error_msg = f"⏱️ 요청 시간이 {timeout_seconds}초를 초과했습니다. 나중에 다시 시도해 주세요."
                logging.error(f"Query timed out after {timeout_seconds} seconds")
                return {"error": error_msg}, error_msg, ""
            except Exception as e:
                import traceback
                error_msg = f"쿼리 처리 중 오류 발생: {str(e)}"
                error_trace = traceback.format_exc()
                logging.error(f"{error_msg}\n{error_trace}")
                return {"error": error_msg}, error_msg, error_trace
                
            final_text = "".join(accumulated_text_obj)
            final_tool = "".join(accumulated_tool_obj)
            
            # If no streaming content was captured, try to extract from response
            if not final_text and response:
                if isinstance(response, dict):
                    if "output" in response:
                        final_text = str(response["output"])
                    elif "content" in response:
                        final_text = str(response["content"])
                    else:
                        final_text = str(response)
                else:
                    final_text = str(response)
                    
                # Update the placeholder with the final text
                text_placeholder.markdown(final_text)
            
            logging.debug(f"Final text length: {len(final_text)}")
            logging.debug(f"Final text: {final_text[:100]}...")
            logging.debug(f"Final tool content length: {len(final_tool)}")
            logging.debug(f"Final tool content: {final_tool[:100]}...")
            return response, final_text, final_tool
        else:
            logging.warning("Agent not initialized before query")
            return (
                {"error": "🚫 에이전트가 초기화되지 않았습니다."},
                "🚫 에이전트가 초기화되지 않았습니다.",
                "",
            )
    except Exception as e:
        import traceback
        error_msg = f"❌ 쿼리 처리 중 오류 발생: {str(e)}"
        error_trace = traceback.format_exc()
        logging.error(f"{error_msg}\n{error_trace}")
        return {"error": error_msg}, error_msg, error_trace


def load_selected_prompt():
    selected = st.session_state["prompt_selectbox"]
    prompts_dict = prompt_data.get("prompts", {})
    if selected in prompts_dict:
        st.session_state.selected_prompt_name = selected
        st.session_state.selected_prompt_text = prompts_dict[selected]["prompt"]
        st.session_state.prompt_loaded = True
        st.session_state["sidebar_edit_prompt_text"] = prompts_dict[selected]["prompt"]


def load_selected_tool():
    selected = st.session_state["tool_selectbox"]
    logging.debug(f"Selected tool: {selected}")
    selected_tool = next((t for t in tools_list if t["name"] == selected), None)
    if selected_tool:
        logging.debug(f"Loading tool configuration from: {selected_tool['path']}")
        try:
            with open(selected_tool["path"], encoding="utf-8") as f:
                st.session_state.tool_config = json.load(f)
            st.session_state.file_path = selected_tool["path"]
            st.session_state.loaded = True
            # Normalize pending MCP config: only keep valid connection fields
            raw_conf = st.session_state.tool_config.get("mcpServers", st.session_state.tool_config)
            pending_conf = {}
            for srv_name, srv_cfg in raw_conf.items():
                if "url" in srv_cfg:
                    # SSE connection
                    conf = {"transport": srv_cfg.get("transport", "sse"), "url": srv_cfg["url"]}
                    if "headers" in srv_cfg:
                        conf["headers"] = srv_cfg["headers"]
                    if "timeout" in srv_cfg:
                        conf["timeout"] = srv_cfg["timeout"]
                    if "sse_read_timeout" in srv_cfg:
                        conf["sse_read_timeout"] = srv_cfg["sse_read_timeout"]
                    if "session_kwargs" in srv_cfg:
                        conf["session_kwargs"] = srv_cfg["session_kwargs"]
                else:
                    # stdio connection
                    conf = {"transport": srv_cfg.get("transport", "stdio"), "command": srv_cfg["command"], "args": srv_cfg["args"]}
                    if "env" in srv_cfg:
                        conf["env"] = srv_cfg["env"]
                    if "cwd" in srv_cfg:
                        conf["cwd"] = srv_cfg["cwd"]
                    if "encoding" in srv_cfg:
                        conf["encoding"] = srv_cfg["encoding"]
                    if "encoding_error_handler" in srv_cfg:
                        conf["encoding_error_handler"] = srv_cfg["encoding_error_handler"]
                    if "session_kwargs" in srv_cfg:
                        conf["session_kwargs"] = srv_cfg["session_kwargs"]
                pending_conf[srv_name] = conf
            # Store direct mapping for initialization (initialize_session will unpack it)
            st.session_state.pending_mcp_config = pending_conf
            logging.debug("Tool configuration loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading tool configuration: {str(e)}")


# 세션 상태 초기화
if "session_initialized" not in st.session_state:
    logging.debug('Session state not initialized, setting default values')
    st.session_state.session_initialized = False  # 세션 초기화 상태 플래그
    st.session_state.agent = None  # ReAct 에이전트 객체 저장 공간
    st.session_state.agent_type = None  # 에이전트 타입 (pandas 또는 langgraph)
    st.session_state.history = []  # 대화 기록 저장 리스트
    st.session_state.mcp_client = None  # MCP 클라이언트 객체 저장 공간
    st.session_state.timeout_seconds = 180  # 응답 생성 제한 시간(초), 기본값 120초
    st.session_state.selected_model = "gpt-4o"  # 기본 모델 선택
    st.session_state.recursion_limit = 100  # 재귀 호출 제한, 기본값 100
    st.session_state.selected_prompt_text = ""  # initialize selected prompt text
    st.session_state.temperature = 0.1  # 기본 temperature 설정
    st.session_state.dataframe = None          # 🆕 DataFrame 보관용
    st.session_state.pending_tool_calls = []  # 대기 중인 도구 호출 목록
    st.session_state.tool_responses = {}  # 도구 응답 저장 공간
    st.session_state.current_message_visualizations = []  # 🆕 현재 메시지 시각화 저장
    # Load default system prompt if none selected
    try:
        with open("prompts/system_prompt.yaml", "r", encoding="utf-8") as f:
            sys_data = yaml.safe_load(f)
            default_prompt = sys_data.get("template", "")
            # store system prompt separately for tool usage and initialize selected prompt
            st.session_state.system_prompt_text = default_prompt
            st.session_state.selected_prompt_text = default_prompt
    except Exception as e:
        logging.warning(f"Failed to load system prompt: {e}")

    # Auto-load AI App settings from URL 'id' param
    query_params = st.query_params
    if "id" in query_params:
        app_id = query_params["id"]
        if not st.session_state.get("auto_loaded", False):
            try:
                with open("store/ai_app_store.json", "r", encoding="utf-8") as f:
                    ai_app_store = json.load(f)
                app_found = False
                for section in ai_app_store.get("AIAppStore", []):
                    if app_found:
                        continue
                    for app in section.get("apps", []):
                        url_parts = urlsplit(app.get("url", ""))
                        params = parse_qs(url_parts.query)
                        if params.get("id", [None])[0] == app_id:
                            st.session_state.selected_model = app.get("model", st.session_state.selected_model)
                            st.session_state.temperature = app.get("temperature", st.session_state.temperature)
                            prompt_text = app.get("prompt", "")
                            if prompt_text:
                                st.session_state.selected_prompt_text = prompt_text
                                st.session_state.sidebar_edit_prompt_text = prompt_text
                            tool_config = app.get("tools", {})
                            st.session_state.tool_config = tool_config
                            raw_conf = tool_config.get("mcpServers", tool_config)
                            pending_conf = {}
                            for srv_name, srv_cfg in raw_conf.items():
                                if "url" in srv_cfg:
                                    conf = {"transport": srv_cfg.get("transport", "sse"), "url": srv_cfg["url"]}
                                    for k in ["headers", "timeout", "sse_read_timeout", "session_kwargs"]:
                                        if k in srv_cfg:
                                            conf[k] = srv_cfg[k]
                                else:
                                    conf = {"transport": srv_cfg.get("transport", "stdio"), "command": srv_cfg.get("command"), "args": srv_cfg.get("args")}
                                    for k in ["env", "cwd", "encoding", "encoding_error_handler", "session_kwargs"]:
                                        if k in srv_cfg:
                                            conf[k] = srv_cfg[k]
                                pending_conf[srv_name] = conf
                            st.session_state.pending_mcp_config = pending_conf
                            st.session_state.auto_loaded = True
                            st.session_state.prompt_loaded = True
                            st.session_state.prompt_selectbox = ""
                            st.session_state.tool_selectbox = ""
                            st.session_state.loaded = True
                            st.session_state.app_title = app.get("title", "Universal Agent")
                            success = st.session_state.event_loop.run_until_complete(initialize_session(st.session_state.pending_mcp_config))
                            st.session_state.session_initialized = success
                            if success:
                                st.rerun()
                            app_found = True
                            break
                    if app_found:
                        break
            except Exception as e:
                st.error(f"Error loading AI App config: {e}")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()

try:
    # Suppress async generator cleanup errors
    sys.set_asyncgen_hooks(finalizer=lambda agen: None)
except AttributeError as e:
    logging.error(f'AttributeError: {str(e)}')


# Load MCP config JSON paths for tools selection
MCP_CONFIG_DIR = "mcp-config"
os.makedirs(MCP_CONFIG_DIR, exist_ok=True)
json_paths = glob.glob(f"{MCP_CONFIG_DIR}/*.json")
if not json_paths and not os.path.exists(f"{MCP_CONFIG_DIR}/mcp_config.json"):
    default_config = {"mcpServers": {}}
    with open(f"{MCP_CONFIG_DIR}/mcp_config.json", "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=2, ensure_ascii=False)
    json_paths = [f"{MCP_CONFIG_DIR}/mcp_config.json"]

st.sidebar.markdown("##### 💡 Store에서 장바구니에 담은 Prompt와 MCP Tool을 조합하여 나만의 AI Agent를 만들어 보세요.")

# --- Prompt Store (프롬프트 선택 및 관리) ---
# EMP_NO 기반 프롬프트 경로 설정
PROMPT_CONFIG_DIR = "prompt-config"
logging.debug('Loading configuration from .env')
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
EMP_NO = os.getenv("EMP_NO", "default_emp_no")
EMP_NAME = os.getenv("EMP_NAME", "default_emp_name")
PROMPT_STORE_PATH = os.path.join(PROMPT_CONFIG_DIR, f"{EMP_NO}.json")

# 프롬프트 파일이 없으면 안내 메시지 출력
if not os.path.exists(PROMPT_STORE_PATH):
    st.sidebar.warning(f"{PROMPT_STORE_PATH} 파일이 없습니다. Prompt Store에서 장바구니 저장을 먼저 해주세요.")
    prompt_data = {"prompts": {}}
else:
    with open(PROMPT_STORE_PATH, encoding="utf-8") as f:
        prompt_data = json.load(f)


# --- Sidebar for File Selection, Save, and Tool List ---
with st.sidebar:
    
    st.selectbox(
        "모델 선택",
        options=list(OUTPUT_TOKEN_INFO.keys()),
        key="selected_model",
    )
    st.slider(
        "Temperature",
        min_value=0.0,
        max_value=0.5,
        step=0.01,
        key="temperature",
        help="Temperature: 낮을수록 고정된 답변, 높을수록 창의적인 답변",
    )
    
    # 🆕 CSV 업로드 섹션을 더 눈에 띄게 수정
    st.markdown("---")
    st.markdown("### 📊 데이터 분석")
    uploaded_csv = st.file_uploader("📂 CSV 파일 업로드", type=["csv"], help="데이터 분석을 위한 CSV 파일을 업로드하세요")
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            st.session_state.dataframe = df
            st.success(f"✅ CSV 로드 완료!")
            
            # 🆕 데이터 정보 표시
            with st.expander("📈 데이터 정보", expanded=True):
                st.write(f"**행**: {df.shape[0]:,}")
                st.write(f"**열**: {df.shape[1]:,}")
                st.write(f"**컬럼**: {', '.join(df.columns.tolist()[:5])}" + ("..." if len(df.columns) > 5 else ""))
                
                # 샘플 데이터 미리보기
                st.write("**미리보기:**")
                st.dataframe(df.head(3), use_container_width=True)
            
            # 🆕 Agent가 생성되면 자동 분석 버튼 안내
            if (st.session_state.get("session_initialized", False) and 
                st.session_state.get("agent_type") == "pandas"):
                st.info("💡 아래 '🤖 자동 데이터 분석' 섹션에서 '🚀 자동 데이터 분석 시작' 버튼을 클릭하여 데이터 분석을 시작하세요!")
                
        except Exception as e:
            st.session_state.dataframe = None
            st.error(f"CSV 로드 실패: {e}")
    elif st.session_state.dataframe is not None:
        # 이미 로드된 데이터가 있으면 정보 표시
        df = st.session_state.dataframe
        st.info(f"✅ 데이터 로드됨: {df.shape[0]:,} × {df.shape[1]:,}")
    
    st.markdown("---")
    
    # 🆕 자동 데이터 분석 섹션 (pandas agent가 있고 데이터가 로드된 경우에만 표시)
    if (st.session_state.get("session_initialized", False) and 
        st.session_state.get("agent_type") == "pandas" and
        st.session_state.dataframe is not None):
        
        st.markdown("### 🤖 자동 데이터 분석")
        
        with st.expander("🚀 자동 분석 기능", expanded=True):
            st.write("""
            **📊 분석 항목:**
            - 데이터 개요 및 구조 분석
            - 컬럼 타입별 분류 및 시각화  
            - 결측값 현황 분석
            - 맞춤형 분석 방향 제안
            - 우선순위 기반 분석 가이드
            """)
            
            if st.button("🚀 자동 데이터 분석 시작", use_container_width=True, key="auto_analysis_start"):
                # 자동 분석 실행
                success = auto_analyze_and_greet(st.session_state.dataframe)
                if success and "auto_analysis_result" in st.session_state:
                    # 분석 결과를 history에 추가
                    analysis_result = st.session_state.auto_analysis_result
                    message_data = {
                        "role": "assistant",
                        "content": analysis_result["content"],
                        "followups": analysis_result["followups"]
                    }
                    if analysis_result["visualizations"]:
                        message_data["visualizations"] = analysis_result["visualizations"]
                    st.session_state.history.append(message_data)
                    del st.session_state.auto_analysis_result
                    st.success("✅ 자동 데이터 분석이 완료되었습니다!")
                    st.rerun()
                else:
                    st.error("❌ 자동 분석 중 오류가 발생했습니다.")
        
        # 🆕 현재 데이터 재분석 버튼
        if st.button("🔄 현재 데이터 재분석", use_container_width=True, key="reanalyze_data"):
            # 재분석 실행
            success = auto_analyze_and_greet(st.session_state.dataframe)
            if success and "auto_analysis_result" in st.session_state:
                # 분석 결과를 history에 추가
                analysis_result = st.session_state.auto_analysis_result
                message_data = {
                    "role": "assistant",
                    "content": analysis_result["content"],
                    "followups": analysis_result["followups"]
                }
                if analysis_result["visualizations"]:
                    message_data["visualizations"] = analysis_result["visualizations"]
                st.session_state.history.append(message_data)
                del st.session_state.auto_analysis_result
                st.success("✅ 데이터 재분석이 완료되었습니다!")
                st.rerun()
            else:
                st.error("❌ 재분석 중 오류가 발생했습니다.")

    st.markdown("---")

    prompts_dict = prompt_data.get("prompts", {})
    prompt_names = list(prompts_dict.keys()) if prompts_dict else ["(등록된 프롬프트 없음)"]
    if st.session_state.get("auto_loaded", False):
        prompt_names = [""] + prompt_names
    st.selectbox(
        "프롬프트 선택",
        prompt_names,
        key="prompt_selectbox",
        on_change=load_selected_prompt,
    )
    if st.session_state.get("auto_loaded", False):
        new_prompt_text = st.text_area("프롬프트 내용", key="sidebar_edit_prompt_text", height=120)
    else:
        # Ensure selected prompt loaded on initial render
        if "prompt_loaded" not in st.session_state or not st.session_state.prompt_loaded:
            load_selected_prompt()

    # 프롬프트 선택 시 바로 아래에 내용 보여주고 수정/저장 가능하게
    selected_prompt = st.session_state.get("prompt_selectbox")
    if selected_prompt and selected_prompt in prompts_dict:
        # Ensure initial sidebar prompt text is set, then bind text_area to session state without default value
        if "sidebar_edit_prompt_text" not in st.session_state:
            st.session_state.sidebar_edit_prompt_text = prompts_dict[selected_prompt]["prompt"]
        new_prompt_text = st.text_area("프롬프트 내용", key="sidebar_edit_prompt_text", height=120)
        if "share_mode" not in st.session_state:
            st.session_state.share_mode = False
        if "clear_share_prompt_title" not in st.session_state:
            st.session_state.clear_share_prompt_title = False
        if st.button("📤 프롬프트 공유", key="sidebar_share_prompt", use_container_width=True):
            st.session_state.share_mode = True
            if st.session_state.share_mode:
                new_title = st.text_input(
                    "공유할 프롬프트 제목을 입력하세요",
                    key="share_prompt_title",
                    value="" if st.session_state.clear_share_prompt_title else st.session_state.get("share_prompt_title", "")
                )
                if st.button("공유", key="share_prompt_confirm", use_container_width=True):
                    global_prompt_store_path = os.path.join("store", "prompt_store.json")
                    if os.path.exists(global_prompt_store_path):
                        with open(global_prompt_store_path, encoding="utf-8") as f:
                            global_prompt_data = json.load(f)
                    else:
                        global_prompt_data = {"prompts": {}}
                    global_prompts_dict = global_prompt_data.get("prompts", {})
                    if not new_title.strip():
                        st.warning("제목을 입력해주세요.")
                    elif new_title in global_prompts_dict:
                        st.warning(f"이미 존재하는 제목입니다: {new_title}. 다른 제목을 입력해주세요.")
                    else:
                        global_prompt_data["prompts"][new_title] = {"prompt": new_prompt_text, "EMP_NO": EMP_NO, "EMP_NAME": EMP_NAME}
                        with open(global_prompt_store_path, "w", encoding="utf-8") as f:
                            json.dump(global_prompt_data, f, indent=2, ensure_ascii=False)
                        st.session_state.saved_msg = f"{new_title} 프롬프트 공유 완료"
                        st.session_state.share_mode = False
                        st.session_state.clear_share_prompt_title = True
                        st.rerun()
                else:
                    st.session_state.clear_share_prompt_title = False
            else:
                st.session_state.clear_share_prompt_title = False


    tools_list = [{"name": Path(p).stem, "path": p} for p in json_paths]
    tool_names = [t["name"] for t in tools_list]
    default_tool_index = 0
    if "file_path" in st.session_state:
        current_name = Path(st.session_state.file_path).stem
        if current_name in tool_names:
            default_tool_index = tool_names.index(current_name)
    if st.session_state.get("auto_loaded", False):
        tool_names = [""] + tool_names
        default_tool_index = 0
    st.selectbox(
        "MCP Tool 목록 선택",
        tool_names,
        key="tool_selectbox",
        index=default_tool_index,
        on_change=load_selected_tool,
    )
    # Load default tool configuration on initial render if not already loaded
    if not st.session_state.get("auto_loaded", False) and not st.session_state.get("loaded", False):
        load_selected_tool()

    # Tool 목록 (List & Delete)
    if st.session_state.get("loaded", False):
        mcp = st.session_state.tool_config.get("mcpServers", {})

        st.markdown("MCP Tool 목록")
        if not mcp:
            st.warning("등록된 도구가 없습니다.")
        else:
            for name in list(mcp.keys()):
                st.write(f"• {name}")


    # 에이전트 설정 적용 및 대화 초기화 버튼 추가
    if st.button("Agent 생성하기", key="create_agent_button", type="primary", use_container_width=True):
        apply_status = st.empty()
        with apply_status.container():
            st.warning("🔄 에이전트를 생성 중입니다... 잠시만 기다려주세요.")
            progress_bar = st.progress(0)
            # 세션 초기화
            st.session_state.session_initialized = False
            st.session_state.agent = None
            progress_bar.progress(30)
            # 초기화 실행
            success = st.session_state.event_loop.run_until_complete(
                initialize_session(st.session_state.pending_mcp_config)
            )
            progress_bar.progress(100)
            if success:
                st.success("✅ 에이전트가 생성되었습니다.")
            else:
                st.error("❌ 에이전트 생성에 실패하였습니다.")
            # 에이전트 초기화 완료 상태를 강제로 설정하여 채팅을 활성화합니다
            st.session_state.session_initialized = True
        # 페이지 새로고침
        st.rerun()
    if st.button("💬 대화 초기화", key="reset_chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.thread_id = random_uuid()
        # 🆕 시각화 데이터도 함께 초기화
        if "current_message_visualizations" in st.session_state:
            st.session_state.current_message_visualizations = []
        # 🆕 matplotlib figures 정리 (메모리 관리)
        try:
            import matplotlib.pyplot as plt
            plt.close('all')  # 모든 figure 닫기
            # 시각화 컨테이너 참조 정리
            if hasattr(st, '_visualization_container'):
                st._visualization_container = None
        except:
            pass
        st.rerun()
    # Add AI App Store registration UI
    if st.button("AI App Store 등록", key="aiapp_register_sidebar", use_container_width=True):
        st.session_state.show_aiapp_registration = True
        # Reset AI App form fields when opening the form
        st.session_state.aiapp_new_name = ""
        st.session_state.aiapp_new_desc = ""
        st.session_state.aiapp_new_url = ""
        st.session_state.aiapp_new_hash_tags = ""
        # Remove any previous upload to avoid session_state conflict
        st.session_state.pop("aiapp_icon_upload", None)

    if st.session_state.get("show_aiapp_registration", False):
        st.subheader("신규 AI App 등록")
        aiapp_name = st.text_input("App 이름", value=st.session_state.get("aiapp_new_name", ""), key="aiapp_new_name")
        aiapp_desc = st.text_input("App 설명", value=st.session_state.get("aiapp_new_desc", ""), key="aiapp_new_desc")
        # Optional icon upload
        aiapp_icon_file = st.file_uploader("App 아이콘 (선택, PNG/JPG)", type=["png","jpg","jpeg"], key="aiapp_icon_upload")
        # URL input with validation
        aiapp_url = st.text_input("App ID (예: Search_Agent)", value=st.session_state.get("aiapp_new_url", ""), key="aiapp_new_url")
        # Optional hashtags input
        aiapp_hash_tags = st.text_input("Hash Tags (콤마로 구분)", value=st.session_state.get("aiapp_new_hash_tags", ""), key="aiapp_new_hash_tags")
        if st.button("등록", key="aiapp_submit_btn", use_container_width=True):
            AI_APP_STORE_PATH = "store/ai_app_store.json"
            # Load existing AI App store
            if not os.path.exists(AI_APP_STORE_PATH) or os.path.getsize(AI_APP_STORE_PATH) == 0:
                apps_by_type = {"auto": [], "user": []}
            else:
                with open(AI_APP_STORE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                apps_by_type = {"auto": [], "user": []}
                for section in data.get("AIAppStore", []):
                    t = section.get("type")
                    if t in apps_by_type:
                        apps_by_type[t].extend(section.get("apps", []))
            # Gather inputs
            name = st.session_state.aiapp_new_name.strip()
            desc = st.session_state.aiapp_new_desc.strip()
            url = f"{URL_BASE}{st.session_state.aiapp_new_url.strip()}"
            icon_file = aiapp_icon_file
            tags_raw = st.session_state.aiapp_new_hash_tags.strip()
            hash_tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
            # Check duplicates
            existing_names = [app.get("title", "") for app in apps_by_type["auto"] + apps_by_type["user"]]
            existing_urls = [app.get("url", "") for app in apps_by_type["auto"] + apps_by_type["user"] if app.get("url")]
            # Validate mandatory fields
            if not name or not desc or not url:
                st.error("이름, 설명, URL을 모두 입력하세요.")
            elif name in existing_names:
                st.error(f"이미 존재하는 App 이름입니다. (중복: {name})")
            else:
                # Validate URL format and characters
                parsed = urlsplit(url)
                if not parsed.scheme or not parsed.netloc:
                    st.error("유효한 URL을 입력하세요. 예: http://example.com")
                elif any(c in url for c in ['"', "'", '<', '>', ' ']):
                    st.error("URL에 허용되지 않는 특수문자가 포함되어 있습니다.")
                elif url in existing_urls:
                    st.error(f"이미 등록된 URL입니다: {url}")
                else:
                    # Save icon if provided
                    icon_name = None
                    if icon_file:
                        ext = Path(icon_file.name).suffix
                        icon_name = f"aiapp_{name}{ext}"
                        icon_dir = os.path.join(ASSETS_DIR, "icons")
                        os.makedirs(icon_dir, exist_ok=True)
                        save_path = os.path.join(icon_dir, icon_name)
                        with open(save_path, "wb") as f:
                            f.write(icon_file.read())
                    # Append new entry
                    apps_by_type["auto"].append({
                        "prompt": st.session_state.selected_prompt_text,
                        "tools": st.session_state.tool_config,
                        "model": st.session_state.selected_model,
                        "temperature": st.session_state.temperature,
                        "title": name,
                        "icon_name": icon_name,
                        "url": url,
                        "hash_tag": hash_tags,
                        "like": 0,
                        "EMP_NO": EMP_NO,
                        "EMP_NAME": EMP_NAME,
                        "description": desc
                    })
                    # Save updated store
                    new_data = {"AIAppStore": [
                        {"type": "auto", "apps": apps_by_type["auto"]},
                        {"type": "user", "apps": apps_by_type["user"]}
                    ]}
                    with open(AI_APP_STORE_PATH, "w", encoding="utf-8") as f:
                        json.dump(new_data, f, indent=2, ensure_ascii=False)
                    st.success("신규 AI App이 등록되었습니다! 새로고침 후 확인하세요.")
                    st.session_state.show_aiapp_registration = False


# --- Main Area ---
title_text = f"🤖 {st.session_state.get('app_title', 'Universal Agent')}"
st.title(title_text)
st.markdown("---")

# 하단 저장 메시지 출력
with st.sidebar:
    if st.session_state.get("saved_msg"):
        st.success(st.session_state.pop("saved_msg"))

# Monkey-patch MultiServerMCPClient.__aexit__ to suppress 'no running event loop' errors during cleanup
_orig_mcp_aexit = MultiServerMCPClient.__aexit__
async def _safe_mcp_aexit(self, exc_type, exc_val, exc_tb):
    try:
        await _orig_mcp_aexit(self, exc_type, exc_val, exc_tb)
    except RuntimeError:
        # Suppress errors when event loop is closed
        pass
    except Exception:
        # Suppress any cleanup-related errors
        pass
MultiServerMCPClient.__aexit__ = _safe_mcp_aexit

# Monkey-patch stdio_client to suppress 'no running event loop' errors
_orig_stdio_client = _stdio.stdio_client
@asynccontextmanager
async def safe_stdio_client(server, errlog=sys.stderr):
    try:
        async with _orig_stdio_client(server, errlog) as (read_stream, write_stream):
            yield read_stream, write_stream
    except RuntimeError:
        # Suppress errors when event loop is closed
        pass
    except Exception:
        # Suppress any cleanup-related errors
        pass
# Override stdio_client with the safe version
_stdio.stdio_client = safe_stdio_client

# Also patch the imported stdio_client in langchain_mcp_adapters.client
import langchain_mcp_adapters.client as _lcmcp_client
_lcmcp_client.stdio_client = safe_stdio_client

# --- Main Chat Area ---
if not st.session_state.session_initialized:
    # 🆕 초기화 메시지를 더 상세하게 수정
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("⚠️ **AI Data Agent가 초기화되지 않았습니다**")
        st.markdown("""
        **시작하기:**
        1. 왼쪽 사이드바에서 **CSV 파일**을 업로드하세요 📂
        2. **Prompt와 MCP Tool**을 선택하세요 (선택사항)
        3. **'Agent 생성하기'** 버튼을 클릭하여 초기화하세요 🚀
        4. 사이드바에 나타나는 **'🚀 자동 데이터 분석 시작'** 버튼 클릭! 📊
        
        💡 **특별 기능**: 
        - CSV 업로드 시 **수동 데이터 분석** 및 **시각화** 제공
        - **탐색적 데이터 분석(EDA)** 가이드 제공
        - **plt.show()** 자동 Streamlit 변환으로 시각화 영구 보존
        - **분석 방향 추천** 및 **맞춤형 후속 질문** 제안
        
        🎯 **추천 시나리오**:
        1. CSV 파일 업로드 → Agent 생성
        2. 사이드바 자동 데이터 분석 버튼 클릭 → 분석 결과 확인
        3. 제안된 분석 방향 중 관심 있는 것 선택  
        4. 대화형으로 심화 분석 진행
        """)
else:
    # 🆕 초기화 완료 후 데이터가 없는 경우 안내
    if st.session_state.dataframe is None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.success("✅ **AI Data Agent가 준비되었습니다!**")
            st.markdown("""
            **다음 단계:**
            1. 왼쪽 사이드바에서 **CSV 파일을 업로드**하세요 📂
            2. 업로드 완료 후 사이드바에 **'🚀 자동 데이터 분석 시작'** 버튼이 나타납니다
            3. 버튼을 클릭하여 **자동 분석**을 시작하세요! 📊
            
            💬 **또는 바로 질문하기:**
            - "안녕하세요!"
            - "어떤 분석이 가능한가요?"
            - "데이터 분석 예시를 보여주세요"
            """)



# --- 대화 기록 출력 ---
print_message()

# --- 사용자 입력 및 처리 ---
user_query = st.session_state.pop("user_query", None) or st.chat_input("💬 질문을 입력하세요")
if user_query:
    if st.session_state.session_initialized:
        st.chat_message("user", avatar="🧑🏻").write(user_query)
        with st.chat_message("assistant", avatar="🤖"):
            # 🆕 컨테이너를 명확히 분리하여 시각화 보존
            response_container = st.container()
            visualization_container = st.container()  
            tool_container = st.container()
            progress_container = st.container()
            
            # 🆕 시각화 컨테이너를 전역으로 설정
            st._visualization_container = visualization_container
            
            with response_container:
                text_placeholder = st.empty()
            
            with tool_container:
                tool_placeholder = st.empty()
                
            with progress_container:
                progress_placeholder = st.empty()
            
            with progress_placeholder.container():
                st.write("🔍 Agent가 도구를 통해 답변을 찾고 있습니다...")
                progress_bar = st.progress(0)
                progress_bar.progress(10)
            
            # 실제 쿼리 처리
            resp, final_text, final_tool = (
                st.session_state.event_loop.run_until_complete(
                    process_query(
                        user_query,
                        text_placeholder,
                        tool_placeholder,
                        st.session_state.timeout_seconds,
                    )
                )
            )
            # 🆕 progress 정리 시 시각화 컨테이너는 건드리지 않음
            progress_placeholder.empty()
            
            # 🆕 시각화 컨테이너 정리 해제
            st._visualization_container = None
            
        if "error" in resp:
            st.error(resp["error"])
        else:
            st.session_state.history.append({"role": "user", "content": user_query})
            
            # 🆕 현재 메시지의 시각화들을 포함해서 저장
            message_data = {
                "role": "assistant", 
                "content": final_text
            }
            
            # 현재 메시지에서 생성된 시각화들이 있으면 포함
            if ("current_message_visualizations" in st.session_state and 
                st.session_state.current_message_visualizations):
                message_data["visualizations"] = st.session_state.current_message_visualizations.copy()
                # 다음 메시지를 위해 초기화
                st.session_state.current_message_visualizations = []
            
            # followup 생성
            openai_api_key = os.getenv("OPENAI_API_KEY", "")
            raw_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com")
            parsed = urlsplit(raw_base)
            openai_api_base = f"{parsed.scheme}://{parsed.netloc}/v1"
            followup_llm = get_followup_llm(
                st.session_state.selected_model,
                0.3,
                openai_api_key,
                openai_api_base,
            )
            followups = st.session_state.event_loop.run_until_complete(
                generate_followups(followup_llm, final_text)
            )
            message_data["followups"] = followups
            
            # 메시지를 history에 추가
            st.session_state.history.append(message_data)
            
            if final_tool and final_tool.strip():
                st.session_state.history.append(
                    {"role": "assistant_tool", "content": final_tool}
                )
            st.rerun()
    else:
        st.warning(
            "⚠️ MCP 서버와 Agent가 초기화되지 않았습니다. 왼쪽 사이드바에서 Prompt와 MCP Tool을 선택하고 'Agent 생성하기' 버튼을 클릭하여 초기화해주세요."
        )