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

# 🆕 기본 데이터 분석 패키지들 import
import numpy as np
import scipy
import seaborn as sns
import sklearn
from sklearn import datasets, metrics, model_selection, preprocessing, linear_model, ensemble, cluster
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 🆕 scikit-learn 추가 모듈들
try:
    from sklearn import neural_network, svm, tree, naive_bayes, neighbors, decomposition, feature_selection, pipeline, mixture
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
    from sklearn.semi_supervised import LabelPropagation, LabelSpreading
    from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.dummy import DummyClassifier, DummyRegressor
    SKLEARN_EXTENDED_AVAILABLE = True
except ImportError:
    SKLEARN_EXTENDED_AVAILABLE = False

# 🆕 딥러닝 프레임워크
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# 🆕 고급 부스팅 모델들
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# 🆕 시계열 분석
try:
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# 🆕 모델 해석 도구들
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# 🆕 이미지 처리
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# 🆕 IPython.display 지원
try:
    from IPython.display import display, HTML, Image as IPyImage, Markdown as IPyMarkdown
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

# 🆕 자연어 처리
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# 🆕 추가 시각화 도구들
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

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
    page_title="AI Data Scientist Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTPUT_TOKEN_INFO = {
    "o4-mini": {"max_tokens": 16000},
    "gpt-4o": {"max_tokens": 16000},
}

# 🆕 ML/DL 자동화 헬퍼 함수들
def create_data_analysis_environment(df=None):
    """
    머신러닝/딥러닝이 대폭 강화된 데이터 분석 환경을 생성합니다.
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
    
    # 🆕 머신러닝 자동화 헬퍼 함수들
    def auto_detect_problem_type(df, target_col):
        """
        데이터와 타겟 컬럼을 분석하여 문제 유형을 자동 감지합니다.
        
        Returns:
            str: 'classification', 'regression', 'clustering', 'anomaly_detection', 'time_series'
        """
        if target_col is None:
            return 'clustering'
        
        if target_col not in df.columns:
            return 'clustering'
        
        target = df[target_col]
        
        # 시계열 데이터 감지
        if df.index.dtype.kind in 'Mm' or any(col.lower() in ['date', 'time', 'timestamp'] for col in df.columns):
            return 'time_series'
        
        # 수치형 타겟이면서 고유값이 많으면 회귀
        if target.dtype.kind in 'biufc':
            unique_ratio = len(target.unique()) / len(target)
            if unique_ratio > 0.05 or len(target.unique()) > 20:
                return 'regression'
            else:
                return 'classification'
        
        # 범주형 타겟이면 분류
        return 'classification'
    
    def auto_select_models(problem_type, df_size):
        """
        문제 유형과 데이터 크기에 따라 적절한 모델들을 자동 선택합니다.
        """
        models = {}
        
        if problem_type == 'classification':
            if df_size < 1000:
                models.update({
                    'Random Forest': ensemble.RandomForestClassifier(random_state=42),
                    'SVM': svm.SVC(random_state=42),
                    'KNN': neighbors.KNeighborsClassifier(),
                    'Naive Bayes': naive_bayes.GaussianNB(),
                    'Decision Tree': tree.DecisionTreeClassifier(random_state=42)
                })
            elif df_size < 10000:
                models.update({
                    'Random Forest': ensemble.RandomForestClassifier(n_estimators=100, random_state=42),
                    'Gradient Boosting': ensemble.GradientBoostingClassifier(random_state=42),
                    'SVM': svm.SVC(random_state=42),
                    'Logistic Regression': linear_model.LogisticRegression(random_state=42),
                    'MLP': neural_network.MLPClassifier(random_state=42, max_iter=500)
                })
            else:
                models.update({
                    'Random Forest': ensemble.RandomForestClassifier(n_estimators=50, random_state=42),
                    'Logistic Regression': linear_model.LogisticRegression(random_state=42),
                    'SGD Classifier': linear_model.SGDClassifier(random_state=42)
                })
                
            # 🆕 고급 부스팅 모델들 추가
            if XGB_AVAILABLE:
                models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            if LGB_AVAILABLE:
                models['LightGBM'] = lgb.LGBMClassifier(random_state=42, verbose=-1)
            if CATBOOST_AVAILABLE:
                models['CatBoost'] = cb.CatBoostClassifier(random_state=42, verbose=False)
        
        elif problem_type == 'regression':
            if df_size < 1000:
                models.update({
                    'Random Forest': ensemble.RandomForestRegressor(random_state=42),
                    'SVR': svm.SVR(),
                    'KNN': neighbors.KNeighborsRegressor(),
                    'Decision Tree': tree.DecisionTreeRegressor(random_state=42)
                })
            elif df_size < 10000:
                models.update({
                    'Random Forest': ensemble.RandomForestRegressor(n_estimators=100, random_state=42),
                    'Gradient Boosting': ensemble.GradientBoostingRegressor(random_state=42),
                    'Linear Regression': linear_model.LinearRegression(),
                    'Ridge': linear_model.Ridge(random_state=42),
                    'MLP': neural_network.MLPRegressor(random_state=42, max_iter=500)
                })
            else:
                models.update({
                    'Random Forest': ensemble.RandomForestRegressor(n_estimators=50, random_state=42),
                    'Linear Regression': linear_model.LinearRegression(),
                    'SGD Regressor': linear_model.SGDRegressor(random_state=42)
                })
                
            # 🆕 고급 부스팅 모델들 추가
            if XGB_AVAILABLE:
                models['XGBoost'] = xgb.XGBRegressor(random_state=42)
            if LGB_AVAILABLE:
                models['LightGBM'] = lgb.LGBMRegressor(random_state=42, verbose=-1)
            if CATBOOST_AVAILABLE:
                models['CatBoost'] = cb.CatBoostRegressor(random_state=42, verbose=False)
        
        elif problem_type == 'clustering':
            models.update({
                'K-Means': cluster.KMeans(random_state=42),
                'DBSCAN': cluster.DBSCAN(),
                'Agglomerative': cluster.AgglomerativeClustering(),
                'Gaussian Mixture': mixture.GaussianMixture(random_state=42)
            })
        
        return models
    
    def auto_ml_pipeline(df, target_col=None, test_size=0.2, cv_folds=5):
        """
        완전 자동화된 머신러닝 파이프라인을 실행합니다.
        
        Args:
            df: 데이터프레임
            target_col: 타겟 컬럼명 (None이면 비지도학습)
            test_size: 테스트 세트 비율
            cv_folds: 교차검증 폴드 수
            
        Returns:
            dict: 결과 정보
        """
        results = {
            'problem_type': None,
            'preprocessing_info': {},
            'model_results': {},
            'best_model': None,
            'feature_importance': None,
            'recommendations': []
        }
        
        try:
            # 1. 문제 유형 감지
            problem_type = auto_detect_problem_type(df, target_col)
            results['problem_type'] = problem_type
            
            print(f"🔍 감지된 문제 유형: {problem_type}")
            
            # 2. 데이터 전처리
            processed_df = df.copy()
            preprocessing_info = {}
            
            # 결측값 처리
            missing_cols = df.isnull().sum()
            missing_cols = missing_cols[missing_cols > 0]
            if len(missing_cols) > 0:
                print(f"⚠️ 결측값 발견: {len(missing_cols)}개 컬럼")
                for col in missing_cols.index:
                    if df[col].dtype.kind in 'biufc':  # 수치형
                        processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                        preprocessing_info[col] = 'filled_with_median'
                    else:  # 범주형
                        processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
                        preprocessing_info[col] = 'filled_with_mode'
            
            # 범주형 데이터 인코딩
            categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
            if target_col and target_col in categorical_cols:
                categorical_cols = categorical_cols.drop(target_col)
                
            for col in categorical_cols:
                if processed_df[col].nunique() > 10:  # 고유값이 많으면 빈도 기반 인코딩
                    freq_encoding = processed_df[col].value_counts().to_dict()
                    processed_df[col] = processed_df[col].map(freq_encoding)
                    preprocessing_info[col] = 'frequency_encoding'
                else:  # One-hot 인코딩
                    dummies = pd.get_dummies(processed_df[col], prefix=col)
                    processed_df = pd.concat([processed_df.drop(col, axis=1), dummies], axis=1)
                    preprocessing_info[col] = 'one_hot_encoding'
            
            results['preprocessing_info'] = preprocessing_info
            
            if problem_type in ['classification', 'regression']:
                # 지도학습
                if target_col not in processed_df.columns:
                    results['recommendations'].append("타겟 컬럼이 존재하지 않습니다.")
                    return results
                
                X = processed_df.drop(target_col, axis=1)
                y = processed_df[target_col]
                
                # 피처 스케일링
                scaler = preprocessing.StandardScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
                
                # 타겟 인코딩 (분류 문제의 경우)
                if problem_type == 'classification' and y.dtype == 'object':
                    label_encoder = preprocessing.LabelEncoder()
                    y = label_encoder.fit_transform(y)
                
                # 데이터 분할
                X_train, X_test, y_train, y_test = model_selection.train_test_split(
                    X_scaled, y, test_size=test_size, random_state=42, stratify=y if problem_type == 'classification' else None
                )
                
                # 모델 선택 및 학습
                models = auto_select_models(problem_type, len(df))
                model_scores = {}
                
                print(f"🤖 {len(models)}개 모델 학습 및 평가 중...")
                
                for name, model in models.items():
                    try:
                        # 교차검증
                        scoring = 'accuracy' if problem_type == 'classification' else 'neg_mean_squared_error'
                        cv_scores = model_selection.cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
                        
                        # 모델 학습
                        model.fit(X_train, y_train)
                        
                        # 예측 및 평가
                        y_pred = model.predict(X_test)
                        
                        if problem_type == 'classification':
                            score = metrics.accuracy_score(y_test, y_pred)
                            precision = metrics.precision_score(y_test, y_pred, average='weighted')
                            recall = metrics.recall_score(y_test, y_pred, average='weighted')
                            f1 = metrics.f1_score(y_test, y_pred, average='weighted')
                            
                            model_scores[name] = {
                                'cv_score': cv_scores.mean(),
                                'cv_std': cv_scores.std(),
                                'test_accuracy': score,
                                'precision': precision,
                                'recall': recall,
                                'f1_score': f1,
                                'model': model
                            }
                        else:
                            score = metrics.r2_score(y_test, y_pred)
                            rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                            mae = metrics.mean_absolute_error(y_test, y_pred)
                            
                            model_scores[name] = {
                                'cv_score': -cv_scores.mean(),  # MSE의 음수였으므로 다시 음수로
                                'cv_std': cv_scores.std(),
                                'test_r2': score,
                                'rmse': rmse,
                                'mae': mae,
                                'model': model
                            }
                        
                        print(f"✅ {name}: CV Score = {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                        
                    except Exception as e:
                        print(f"❌ {name} 학습 실패: {str(e)}")
                        continue
                
                results['model_results'] = model_scores
                
                # 최고 모델 선택
                if model_scores:
                    best_metric = 'test_accuracy' if problem_type == 'classification' else 'test_r2'
                    best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x][best_metric])
                    results['best_model'] = {
                        'name': best_model_name,
                        'model': model_scores[best_model_name]['model'],
                        'scores': model_scores[best_model_name]
                    }
                    
                    print(f"🏆 최고 성능 모델: {best_model_name}")
                    
                    # 피처 중요도 추출
                    best_model = model_scores[best_model_name]['model']
                    if hasattr(best_model, 'feature_importances_'):
                        importance = pd.DataFrame({
                            'feature': X.columns,
                            'importance': best_model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        results['feature_importance'] = importance
                    elif hasattr(best_model, 'coef_'):
                        importance = pd.DataFrame({
                            'feature': X.columns,
                            'coefficient': np.abs(best_model.coef_.flatten() if len(best_model.coef_.shape) > 1 else best_model.coef_)
                        }).sort_values('coefficient', ascending=False)
                        results['feature_importance'] = importance
            
            elif problem_type == 'clustering':
                # 비지도학습
                X = processed_df.select_dtypes(include=[np.number])
                if X.empty:
                    results['recommendations'].append("수치형 데이터가 없어 클러스터링을 수행할 수 없습니다.")
                    return results
                
                # 피처 스케일링
                scaler = preprocessing.StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # 최적 클러스터 수 찾기
                inertias = []
                silhouette_scores = []
                K_range = range(2, min(11, len(X)//2))
                
                for k in K_range:
                    kmeans = cluster.KMeans(n_clusters=k, random_state=42)
                    cluster_labels = kmeans.fit_predict(X_scaled)
                    inertias.append(kmeans.inertia_)
                    silhouette_scores.append(metrics.silhouette_score(X_scaled, cluster_labels))
                
                # 최적 클러스터 수 선택 (실루엣 점수 기준)
                optimal_k = K_range[np.argmax(silhouette_scores)]
                
                # 클러스터링 수행
                models = auto_select_models(problem_type, len(df))
                models['K-Means'].set_params(n_clusters=optimal_k)
                
                clustering_results = {}
                for name, model in models.items():
                    try:
                        cluster_labels = model.fit_predict(X_scaled)
                        silhouette = metrics.silhouette_score(X_scaled, cluster_labels)
                        clustering_results[name] = {
                            'silhouette_score': silhouette,
                            'n_clusters': len(np.unique(cluster_labels)),
                            'model': model,
                            'labels': cluster_labels
                        }
                        print(f"✅ {name}: Silhouette Score = {silhouette:.4f}")
                    except Exception as e:
                        print(f"❌ {name} 실패: {str(e)}")
                
                results['model_results'] = clustering_results
                
                if clustering_results:
                    best_clustering = max(clustering_results.keys(), key=lambda x: clustering_results[x]['silhouette_score'])
                    results['best_model'] = {
                        'name': best_clustering,
                        'model': clustering_results[best_clustering]['model'],
                        'scores': clustering_results[best_clustering]
                    }
            
            # 추천사항 생성
            results['recommendations'].extend([
                "모델 성능을 더 향상시키려면 피처 엔지니어링을 시도해보세요.",
                "하이퍼파라미터 튜닝으로 성능을 개선할 수 있습니다.",
                "교차검증 결과와 테스트 성능을 비교해보세요."
            ])
            
            return results
            
        except Exception as e:
            results['error'] = str(e)
            print(f"❌ AutoML 파이프라인 실행 중 오류: {str(e)}")
            return results
    
    def create_deep_learning_model(problem_type, input_shape, num_classes=None):
        """
        문제 유형에 따라 적절한 딥러닝 모델을 생성합니다.
        """
        if not TF_AVAILABLE:
            print("❌ TensorFlow가 설치되지 않아 딥러닝 모델을 생성할 수 없습니다.")
            return None
        
        if problem_type == 'classification':
            model = models.Sequential([
                layers.Dense(128, activation='relu', input_shape=(input_shape,)),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
                metrics=['accuracy']
            )
            
        elif problem_type == 'regression':
            model = models.Sequential([
                layers.Dense(128, activation='relu', input_shape=(input_shape,)),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dense(1)
            ])
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
        
        return model
    
    def auto_hyperparameter_tuning(model, X_train, y_train, param_grid=None, cv=5):
        """
        자동 하이퍼파라미터 튜닝을 수행합니다.
        """
        if param_grid is None:
            # 기본 매개변수 그리드
            if hasattr(model, 'n_estimators'):
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30]
                }
            elif hasattr(model, 'C'):
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.1, 1]
                }
            else:
                param_grid = {}
        
        if not param_grid:
            print("⚠️ 하이퍼파라미터 그리드가 정의되지 않았습니다.")
            return model
        
        grid_search = model_selection.GridSearchCV(
            model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"🎯 최적 파라미터: {grid_search.best_params_}")
        print(f"🎯 최적 CV 점수: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def explain_model_predictions(model, X_test, X_train=None):
        """
        SHAP을 사용하여 모델 예측을 해석합니다.
        """
        if not SHAP_AVAILABLE:
            print("❌ SHAP이 설치되지 않아 모델 해석을 수행할 수 없습니다.")
            return None
        
        try:
            # 모델 유형에 따라 적절한 explainer 선택
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model, X_train)
            else:
                explainer = shap.Explainer(model)
            
            shap_values = explainer(X_test[:100])  # 처음 100개 샘플만 사용
            
            # SHAP 시각화
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test[:100], show=False)
            plt.title("SHAP Feature Importance Summary")
            plt.show()
            
            return shap_values
            
        except Exception as e:
            print(f"❌ 모델 해석 중 오류: {str(e)}")
            return None
    
    def generate_ml_report(results):
        """
        머신러닝 분석 결과 보고서를 생성합니다.
        """
        report = f"""
# 🤖 AutoML 분석 보고서

## 📊 문제 유형
**{results['problem_type']}**

## 🔧 데이터 전처리
"""
        
        if results['preprocessing_info']:
            for col, action in results['preprocessing_info'].items():
                report += f"- **{col}**: {action}\n"
        else:
            report += "- 전처리가 필요하지 않았습니다.\n"
        
        report += "\n## 🏆 모델 성능 결과\n"
        
        if results['model_results']:
            for model_name, scores in results['model_results'].items():
                report += f"### {model_name}\n"
                for metric, value in scores.items():
                    if metric != 'model' and isinstance(value, (int, float)):
                        report += f"- **{metric}**: {value:.4f}\n"
                report += "\n"
        
        if results['best_model']:
            report += f"\n## 🥇 최고 성능 모델: {results['best_model']['name']}\n"
        
        if results.get('feature_importance') is not None:
            report += "\n## 📈 피처 중요도 (Top 10)\n"
            top_features = results['feature_importance'].head(10)
            for _, row in top_features.iterrows():
                importance_col = 'importance' if 'importance' in row else 'coefficient'
                report += f"- **{row['feature']}**: {row[importance_col]:.4f}\n"
        
        report += "\n## 💡 추천사항\n"
        for rec in results['recommendations']:
            report += f"- {rec}\n"
        
        return report

    # 🆕 강화된 analysis_env 딕셔너리
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
        
        # 🆕 scikit-learn 전체 모듈들
        "sklearn": sklearn,
        "datasets": datasets,
        "metrics": metrics,
        "model_selection": model_selection,
        "preprocessing": preprocessing,
        "linear_model": linear_model,
        "ensemble": ensemble,
        "cluster": cluster,
        "neural_network": neural_network,
        "svm": svm,
        "tree": tree,
        "naive_bayes": naive_bayes,
        "neighbors": neighbors,
        "decomposition": decomposition,
        "feature_selection": feature_selection,
        "pipeline": pipeline,
        "mixture": mixture,
        
        # 🆕 딥러닝 (TensorFlow/Keras)
        "tf": tf if TF_AVAILABLE else None,
        "tensorflow": tf if TF_AVAILABLE else None,
        "keras": keras if TF_AVAILABLE else None,
        "layers": layers if TF_AVAILABLE else None,
        "models": models if TF_AVAILABLE else None,
        "optimizers": optimizers if TF_AVAILABLE else None,
        "callbacks": callbacks if TF_AVAILABLE else None,
        
        # 🆕 고급 부스팅 모델들
        "xgb": xgb if XGB_AVAILABLE else None,
        "xgboost": xgb if XGB_AVAILABLE else None,
        "lgb": lgb if LGB_AVAILABLE else None,
        "lightgbm": lgb if LGB_AVAILABLE else None,
        "cb": cb if CATBOOST_AVAILABLE else None,
        "catboost": cb if CATBOOST_AVAILABLE else None,
        
        # 🆕 시계열 분석
        "sm": sm if STATSMODELS_AVAILABLE else None,
        "statsmodels": sm if STATSMODELS_AVAILABLE else None,
        "seasonal_decompose": seasonal_decompose if STATSMODELS_AVAILABLE else None,
        "ARIMA": ARIMA if STATSMODELS_AVAILABLE else None,
        
        # 🆕 모델 해석
        "shap": shap if SHAP_AVAILABLE else None,
        
        # 🆕 이미지 처리
        "Image": Image if PIL_AVAILABLE else None,
        "PIL": PIL_AVAILABLE,
        
        # 🆕 IPython.display 지원
        "display": display if IPYTHON_AVAILABLE else None,
        "HTML": HTML if IPYTHON_AVAILABLE else None,
        "IPyImage": IPyImage if IPYTHON_AVAILABLE else None,
        "IPyMarkdown": IPyMarkdown if IPYTHON_AVAILABLE else None,
        
        # 🆕 자연어 처리
        "nltk": nltk if NLTK_AVAILABLE else None,
        
        # 기타 유용한 패키지들
        "warnings": warnings,
        "os": os,
        "sys": sys,
        "json": json,
        "time": time,
        
        # 🆕 자주 사용하는 ML 함수들을 직접 접근 가능하게
        "train_test_split": model_selection.train_test_split,
        "cross_val_score": model_selection.cross_val_score,
        "GridSearchCV": model_selection.GridSearchCV,
        "RandomizedSearchCV": model_selection.RandomizedSearchCV,
        
        # 스케일링
        "StandardScaler": preprocessing.StandardScaler,
        "MinMaxScaler": preprocessing.MinMaxScaler,
        "RobustScaler": preprocessing.RobustScaler,
        "LabelEncoder": preprocessing.LabelEncoder,
        "OneHotEncoder": preprocessing.OneHotEncoder,
        
        # 분류 모델들
        "LogisticRegression": linear_model.LogisticRegression,
        "RandomForestClassifier": ensemble.RandomForestClassifier,
        "GradientBoostingClassifier": ensemble.GradientBoostingClassifier,
        "SVC": svm.SVC,
        "DecisionTreeClassifier": tree.DecisionTreeClassifier,
        "KNeighborsClassifier": neighbors.KNeighborsClassifier,
        "GaussianNB": naive_bayes.GaussianNB,
        "MLPClassifier": neural_network.MLPClassifier,
        
        # 회귀 모델들
        "LinearRegression": linear_model.LinearRegression,
        "Ridge": linear_model.Ridge,
        "Lasso": linear_model.Lasso,
        "RandomForestRegressor": ensemble.RandomForestRegressor,
        "GradientBoostingRegressor": ensemble.GradientBoostingRegressor,
        "SVR": svm.SVR,
        "DecisionTreeRegressor": tree.DecisionTreeRegressor,
        "KNeighborsRegressor": neighbors.KNeighborsRegressor,
        "MLPRegressor": neural_network.MLPRegressor,
        
        # 클러스터링
        "KMeans": cluster.KMeans,
        "DBSCAN": cluster.DBSCAN,
        "AgglomerativeClustering": cluster.AgglomerativeClustering,
        "GaussianMixture": mixture.GaussianMixture,
        
        # 차원 축소
        "PCA": decomposition.PCA,
        "TruncatedSVD": decomposition.TruncatedSVD,
        "LDA": LinearDiscriminantAnalysis,
        
        # 🆕 자동화 헬퍼 함수들
        "auto_detect_problem_type": auto_detect_problem_type,
        "auto_select_models": auto_select_models,
        "auto_ml_pipeline": auto_ml_pipeline,
        "create_deep_learning_model": create_deep_learning_model,
        "auto_hyperparameter_tuning": auto_hyperparameter_tuning,
        "explain_model_predictions": explain_model_predictions,
        "generate_ml_report": generate_ml_report,
        
        # 기존 헬퍼 함수들 유지
        "reset_show": reset_show,
        "force_show": force_show,
        "original_show": original_show,
        "original_clf": original_clf,
        "original_cla": original_cla,
        "original_close": original_close,
        "setup_korean_font": setup_korean_font,
        "check_korean_font": check_korean_font,
        "safe_dataframe_check": safe_dataframe_check,
        "diagnose_data": diagnose_data,
        "safe_plot": safe_plot,
        "get_current_visualizations": get_current_visualizations,
        "clear_current_visualizations": clear_current_visualizations,
        "preview_current_visualizations": preview_current_visualizations,
    }
    
    # DataFrame이 제공된 경우 추가
    if df is not None:
        analysis_env["df"] = df
        analysis_env["data"] = df  # 일반적인 별명도 추가
    
    return analysis_env

# 🆕 강화된 Python Tool 생성 함수
def create_enhanced_python_tool(df=None):
    """
    머신러닝/딥러닝이 대폭 강화된 PythonAstREPLTool을 생성합니다.
    plt.show()가 자동으로 Streamlit에서 동작하도록 패치되어 있습니다.
    
    Args:
        df: 분석할 DataFrame (선택사항)
    
    Returns:
        PythonAstREPLTool: 향상된 Python REPL 도구
    """
    analysis_env = create_data_analysis_environment(df)
    
    # 🆕 강화된 사용자 친화적인 설명과 예제
    description = """
🚀 **지능형 AI Data Scientist 환경**에 오신 것을 환영합니다!

📊 **사전 로드된 패키지들:**

**기본 데이터 분석:**
- 데이터 처리: pandas (pd), numpy (np)
- 시각화: matplotlib (plt), seaborn (sns), streamlit (st)  
- 과학계산: scipy

**🤖 머신러닝 (scikit-learn):**
- 지도학습: linear_model, ensemble, svm, tree, neural_network
- 비지도학습: cluster, decomposition
- 전처리: preprocessing, feature_selection, pipeline
- 평가: metrics, model_selection

**🧠 딥러닝 (TensorFlow/Keras):**
- tensorflow (tf), keras, layers, models, optimizers, callbacks
- 자동 신경망 모델 생성 지원

**⚡ 고급 부스팅 모델들:**
- XGBoost (xgb): 고성능 그래디언트 부스팅
- LightGBM (lgb): 빠른 그래디언트 부스팅
- CatBoost (cb): 범주형 데이터 특화 부스팅

**📈 시계열 분석:**
- statsmodels (sm): ARIMA, 계절성 분해
- seasonal_decompose: 시계열 분해

**🔍 모델 해석:**
- SHAP: 모델 예측 해석 및 피처 중요도

**📷 이미지 처리:**
- PIL/Image: 이미지 로드, 처리, 저장

**📄 IPython Display:**
- display(): 데이터/객체를 깔끔하게 표시
- HTML(): HTML 콘텐츠 렌더링  
- IPyImage: 이미지 표시
- IPyMarkdown: 마크다운 렌더링

**🚀 AutoML 특별 기능들:**
✅ **완전 자동화된 머신러닝 파이프라인** (`auto_ml_pipeline()`)
✅ **지능형 문제 유형 감지** (분류/회귀/클러스터링/시계열 자동 판단)
✅ **데이터 크기별 최적 모델 자동 선택**
✅ **자동 데이터 전처리** (결측값, 범주형 데이터, 스케일링)
✅ **자동 하이퍼파라미터 튜닝** (`auto_hyperparameter_tuning()`)
✅ **딥러닝 모델 자동 생성** (`create_deep_learning_model()`)
✅ **모델 성능 자동 평가 및 비교**
✅ **SHAP 기반 모델 해석** (`explain_model_predictions()`)
✅ **전문가 수준 분석 보고서 생성** (`generate_ml_report()`)

**시각화 특별 기능:**
✅ plt.show() 자동 Streamlit 변환 (시각화 영구 보존)
✅ 도구 호출 정보가 접혀도 시각화는 그대로 유지!
✅ 한글 폰트 자동 설정 및 최적화

🎯 **AutoML 빠른 시작 가이드:**

**1. 완전 자동 분석 (추천!):**
```python
# 타겟 컬럼이 있는 지도학습
results = auto_ml_pipeline(df, target_col='target_column_name')
print(generate_ml_report(results))

# 타겟 없는 클러스터링
results = auto_ml_pipeline(df)
print(generate_ml_report(results))
```

**2. 단계별 커스텀 분석:**
```python
# 문제 유형 자동 감지
problem_type = auto_detect_problem_type(df, 'target_col')
print(f"감지된 문제: {problem_type}")

# 적절한 모델들 자동 선택
models = auto_select_models(problem_type, len(df))

# 최고 모델에 하이퍼파라미터 튜닝 적용
best_model = auto_hyperparameter_tuning(model, X_train, y_train)

# SHAP으로 모델 해석
shap_values = explain_model_predictions(best_model, X_test, X_train)
```

**3. 딥러닝 모델 생성:**
```python
# 문제 유형에 맞는 신경망 자동 생성
model = create_deep_learning_model('classification', input_shape=X.shape[1], num_classes=3)
model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)
```

**4. 고급 부스팅 모델들:**
```python
# XGBoost
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

# LightGBM  
lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)

# CatBoost (범주형 데이터에 강함)
cb_model = cb.CatBoostClassifier(random_state=42, verbose=False)
cb_model.fit(X_train, y_train)
```

**5. IPython Display 활용:**
```python
# 데이터프레임 깔끔하게 표시
display(df.head())

# HTML 테이블로 결과 표시
display(HTML(df.to_html()))

# 분석 결과를 마크다운으로 표시
display(IPyMarkdown("## 분석 결과\\n**정확도**: 95.3%"))
```

**📊 시각화 예시:**
```python
# 상관관계 히트맵 (한글 지원)
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('변수 간 상관관계')
plt.show()  # 자동으로 Streamlit에 표시됩니다!

# 피처 중요도 시각화
if 'feature_importance' in results:
    plt.figure(figsize=(10, 6))
    top_features = results['feature_importance'].head(10)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.title('Top 10 피처 중요도')
    plt.show()
```

**🔥 실전 사용 예시:**

**전체 자동화 워크플로우:**
```python
# 1단계: 데이터 탐색
print("📊 데이터 기본 정보:")
print(df.info())
print("\\n📈 기초 통계:")
print(df.describe())

# 2단계: 자동 ML 파이프라인 실행
print("\\n🤖 AutoML 파이프라인 시작...")
results = auto_ml_pipeline(df, target_col='target', test_size=0.2, cv_folds=5)

# 3단계: 결과 분석 및 보고서 생성
print("\\n📋 분석 보고서:")
print(generate_ml_report(results))

# 4단계: 최고 모델로 예측
if results['best_model']:
    best_model = results['best_model']['model']
    predictions = best_model.predict(X_test)
    print(f"\\n🎯 예측 완료! 정확도: {results['best_model']['scores']['test_accuracy']:.4f}")

# 5단계: 모델 해석 (SHAP)
shap_values = explain_model_predictions(best_model, X_test, X_train)
```

**💡 도움말 및 디버깅:**
```python
# 데이터 진단
diagnose_data(df)

# 한글 폰트 확인
check_korean_font()

# 시각화 상태 확인
print(f"현재 생성된 시각화: {get_current_visualizations()}개")
```

**🎓 학습 가이드:**
- **초보자**: `auto_ml_pipeline(df, 'target_col')` 한 줄로 시작!
- **중급자**: 개별 모델들을 비교하고 튜닝해보세요
- **고급자**: 커스텀 피처 엔지니어링과 앙상블 기법 활용

DataFrame은 'df' 또는 'data' 변수로 접근할 수 있습니다.
무엇을 분석하고 싶으신지 말씀해 주세요! 🤖✨
"""
    
    return PythonAstREPLTool(
        locals=analysis_env,
        description=description,
        name="python_repl_ast",
        handle_tool_error=True
    )

# 🆕 강화된 자동 데이터 분석 및 인사말 생성 함수
def auto_analyze_and_greet(df):
    """
    데이터 로드 시 ML/DL 기능을 활용한 지능적 분석을 수행하고 맞춤형 가이드를 생성합니다.
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
        
        # 🆕 ML/DL 관점에서 데이터 분석
        # 1. 가능한 타겟 컬럼 추정
        potential_targets = []
        for col in columns:
            # 일반적인 타겟 컬럼명 패턴
            if any(keyword in col.lower() for keyword in ['target', 'label', 'class', 'category', 'outcome', 'result', 'status', 'type']):
                potential_targets.append(col)
            # 이진 분류 가능성
            elif df[col].nunique() == 2 and col in categorical_cols:
                potential_targets.append(col)
            # 수치형이지만 클래스처럼 보이는 경우
            elif col in numeric_cols and df[col].nunique() <= 10 and all(df[col].dropna() == df[col].dropna().astype(int)):
                potential_targets.append(col)
        
        # 2. 데이터 복잡도 분석
        data_complexity = "단순"
        if shape[0] > 100000:
            data_complexity = "대용량"
        elif shape[1] > 50:
            data_complexity = "고차원"
        elif len(categorical_cols) > len(numeric_cols):
            data_complexity = "범주형 중심"
        elif missing_values > shape[0] * shape[1] * 0.1:
            data_complexity = "결측값 다량"
            
        # 3. 추천 ML 알고리즘 미리 분석
        recommended_algorithms = []
        if len(numeric_cols) >= 2:
            recommended_algorithms.extend(["선형 회귀", "랜덤 포레스트", "XGBoost"])
        if len(categorical_cols) > 0:
            recommended_algorithms.extend(["결정 트리", "CatBoost"])
        if shape[0] > 10000:
            recommended_algorithms.extend(["딥러닝", "LightGBM"])
        if not potential_targets:
            recommended_algorithms = ["K-Means 클러스터링", "DBSCAN", "계층적 클러스터링"]
            
        # 인사말 및 분석 결과 생성
        greeting_content = f"""🎉 **AI Data Scientist 환경에 오신 것을 환영합니다!**

📊 **로드된 데이터 개요:**
- **데이터 크기**: {shape[0]:,} 행 × {shape[1]:,} 열
- **메모리 사용량**: {memory_usage:.2f} MB
- **데이터 복잡도**: {data_complexity}
- **결측값**: {missing_values:,} 개 ({missing_values/(shape[0]*shape[1])*100:.1f}%)

📋 **컬럼 구성:**
- **수치형 컬럼** ({len(numeric_cols)}개): {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}
- **범주형 컬럼** ({len(categorical_cols)}개): {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}
{'- **날짜형 컬럼** (' + str(len(datetime_cols)) + '개): ' + ', '.join(datetime_cols[:3]) + ('...' if len(datetime_cols) > 3 else '') if datetime_cols else ''}

🤖 **ML/DL 분석 인사이트:**"""

        # 🆕 잠재적 타겟 컬럼 정보
        if potential_targets:
            greeting_content += f"""
🎯 **추정 타겟 컬럼**: {', '.join(potential_targets[:3])}{'...' if len(potential_targets) > 3 else ''}"""
        else:
            greeting_content += f"""
🎯 **분석 유형**: 비지도학습 (클러스터링/패턴 발견) 적합"""

        # 🆕 추천 알고리즘
        greeting_content += f"""
⚡ **추천 ML 알고리즘**: {', '.join(recommended_algorithms[:4])}{'...' if len(recommended_algorithms) > 4 else ''}"""

        # 데이터 미리보기
        greeting_content += f"""

🔍 **데이터 미리보기:**"""
        preview_text = df.head(3).to_string()
        greeting_content += f"\n```\n{preview_text}\n```\n"
        
        # 🆕 ML/DL 관점의 맞춤형 분석 제안
        ml_suggestions = []
        
        # 1. 데이터 품질 분석
        if missing_values > 0:
            missing_ratio = missing_values / (shape[0] * shape[1])
            if missing_ratio > 0.3:
                ml_suggestions.append("📍 **데이터 품질**: 결측값이 30% 이상입니다. 고급 결측값 처리 기법 적용 필요")
            else:
                ml_suggestions.append("📍 **데이터 품질**: 결측값 비율이 적당합니다. 기본 전처리로 해결 가능")
        
        # 2. 모델링 전략 제안
        if potential_targets:
            # 지도학습 시나리오
            target_analysis = []
            for target in potential_targets[:2]:  # 상위 2개만 분석
                target_unique = df[target].nunique()
                if target_unique <= 10:
                    target_analysis.append(f"{target} (분류: {target_unique}개 클래스)")
                else:
                    target_analysis.append(f"{target} (회귀: 연속값)")
            
            ml_suggestions.append(f"🎯 **모델링 전략**: 지도학습 - {', '.join(target_analysis)}")
            ml_suggestions.append("🚀 **AutoML 추천**: `auto_ml_pipeline(df, target_col='타겟컬럼명')` 실행")
        else:
            # 비지도학습 시나리오
            ml_suggestions.append("🎯 **모델링 전략**: 비지도학습 - 패턴 발견 및 클러스터링")
            ml_suggestions.append("🚀 **AutoML 추천**: `auto_ml_pipeline(df)` 실행")
        
        # 3. 데이터 크기별 최적화 제안
        if shape[0] > 100000:
            ml_suggestions.append("⚡ **성능 최적화**: 대용량 데이터 - LightGBM, 샘플링 기법 권장")
        elif shape[0] < 1000:
            ml_suggestions.append("⚠️ **데이터 크기**: 소규모 데이터 - 교차검증 강화, 정규화 기법 필수")
        
        # 4. 피처 엔지니어링 제안
        if len(categorical_cols) > len(numeric_cols):
            ml_suggestions.append("🔧 **피처 엔지니어링**: 범주형 데이터 중심 - 인코딩 최적화, CatBoost 활용")
        if len(numeric_cols) >= 10:
            ml_suggestions.append("📊 **차원 축소**: 고차원 데이터 - PCA, 피처 선택 기법 고려")
        
        # 5. 딥러닝 적용 가능성
        if shape[0] > 10000 and len(numeric_cols) > 5:
            ml_suggestions.append("🧠 **딥러닝**: 충분한 데이터량 - 신경망 모델 적용 가능")
        
        greeting_content += "\n💡 **AI 기반 분석 전략:**\n"
        for i, suggestion in enumerate(ml_suggestions[:5], 1):
            greeting_content += f"{i}. {suggestion}\n"
            
        # 🆕 단계별 실행 가이드
        greeting_content += """
🚀 **단계별 AutoML 가이드:**

**🥇 LEVEL 1 - 완전 자동 분석 (초보자 추천)**
```python
# 원클릭 AutoML - 모든 것을 자동으로!
results = auto_ml_pipeline(df, target_col='타겟컬럼명')  # 지도학습
# 또는
results = auto_ml_pipeline(df)  # 비지도학습
print(generate_ml_report(results))
```

**🥈 LEVEL 2 - 커스텀 분석 (중급자)**
```python
# 문제 유형 분석 후 맞춤 모델링
problem_type = auto_detect_problem_type(df, '타겟컬럼')
models = auto_select_models(problem_type, len(df))
# 개별 모델 학습 및 비교...
```

**🥉 LEVEL 3 - 전문가 분석 (고급자)**
```python
# 딥러닝 + 고급 부스팅 + 모델 해석
dl_model = create_deep_learning_model('classification', X.shape[1], num_classes)
xgb_model = xgb.XGBClassifier()
shap_values = explain_model_predictions(best_model, X_test)
```

**⚡ 빠른 시작 명령어:**
- `auto_ml_pipeline(df, 'target_col')` - 완전 자동 ML
- `df.describe()` - 기초 통계 요약
- `df.hist(figsize=(15, 10)); plt.show()` - 전체 변수 분포
- `sns.heatmap(df.corr(), annot=True); plt.title('상관관계'); plt.show()` - 상관관계 분석
- `auto_detect_problem_type(df, 'target_col')` - 문제 유형 자동 감지

무엇을 분석하고 싶으신지 말씀해 주세요! 🤖✨"""

        # 🆕 고급 시각화 생성
        visualizations = []
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 1. 데이터 타입과 ML 적합성 분석 시각화
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 컬럼 타입 분포
            type_counts = {'수치형': len(numeric_cols), '범주형': len(categorical_cols), '날짜형': len(datetime_cols)}
            type_counts = {k: v for k, v in type_counts.items() if v > 0}
            
            colors = ['#FF9999', '#66B2FF', '#99FF99'][:len(type_counts)]
            ax1.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%', startangle=90, colors=colors)
            ax1.set_title('컬럼 타입 분포', fontsize=14, fontweight='bold')
            
            # 결측값 현황
            if missing_values > 0 and len(missing_cols) > 0:
                missing_data = pd.Series(missing_cols).head(10)
                bars = ax2.bar(range(len(missing_data)), missing_data.values, color='coral', alpha=0.7)
                ax2.set_title('결측값 현황 (Top 10)', fontsize=14, fontweight='bold')
                ax2.set_ylabel('결측값 개수')
                ax2.set_xticks(range(len(missing_data)))
                ax2.set_xticklabels(missing_data.index, rotation=45, ha='right')
                
                # 값 라벨 추가
                for bar, value in zip(bars, missing_data.values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(missing_data.values)*0.01, 
                            f'{int(value)}', ha='center', va='bottom', fontsize=10)
            else:
                ax2.text(0.5, 0.5, f'✅ 결측값 없음!\n완벽한 데이터셋', 
                        ha='center', va='center', transform=ax2.transAxes, 
                        fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
                ax2.set_title('🎉 데이터 품질 현황', fontsize=14, fontweight='bold')
                ax2.axis('off')
            
            # 🆕 ML 알고리즘 적합성 차트
            if recommended_algorithms:
                algo_scores = {}
                # 간단한 점수 시스템
                base_score = 0.7
                if shape[0] > 10000: base_score += 0.1
                if len(numeric_cols) > 5: base_score += 0.1  
                if missing_values == 0: base_score += 0.1
                
                for algo in recommended_algorithms[:6]:
                    score = base_score + np.random.uniform(-0.1, 0.1)  # 약간의 변동
                    algo_scores[algo] = min(score, 1.0)
                
                algos = list(algo_scores.keys())
                scores = list(algo_scores.values())
                colors_algo = plt.cm.viridis(np.linspace(0, 1, len(algos)))
                
                bars = ax3.barh(algos, scores, color=colors_algo, alpha=0.8)
                ax3.set_xlim(0, 1)
                ax3.set_xlabel('적합성 점수')
                ax3.set_title('추천 ML 알고리즘', fontsize=14, fontweight='bold')
                
                # 점수 라벨 추가
                for bar, score in zip(bars, scores):
                    ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                            f'{score:.2f}', va='center', fontsize=10)
            else:
                ax3.text(0.5, 0.5, '🔍 분석 중...\n데이터 탐색 필요', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                ax3.set_title('ML 알고리즘 분석', fontsize=14, fontweight='bold')
                ax3.axis('off')
            
            # 🆕 데이터 복잡도 및 권장 접근법
            complexity_info = {
                '데이터 크기': f'{shape[0]:,} × {shape[1]:,}',
                '복잡도': data_complexity,
                '메모리': f'{memory_usage:.1f} MB',
                '품질 점수': f'{((shape[0]*shape[1] - missing_values)/(shape[0]*shape[1])*100):.1f}%'
            }
            
            y_pos = np.arange(len(complexity_info))
            info_text = '\n'.join([f'{k}: {v}' for k, v in complexity_info.items()])
            
            ax4.text(0.5, 0.7, '데이터 요약', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=16, fontweight='bold')
            ax4.text(0.5, 0.3, info_text, ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
            ax4.axis('off')
            
            # 🔧 수정된 부분: 레이아웃 조정을 먼저 하고 타이틀 추가
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 위쪽과 아래쪽에 여백 확보
            
            # 🔧 이모지 대신 텍스트로 변경하여 호환성 개선
            plt.suptitle('AI Data Scientist - 데이터 분석 대시보드', 
                         fontsize=16, fontweight='bold', y=0.97)  # y 위치를 0.97로 조정
            
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
            logging.warning(f"고급 시각화 생성 실패: {viz_error}")
        
        # 🆕 맞춤형 후속 질문 생성
        followups = [
            "🚀 auto_ml_pipeline(df, 'target_col') 실행해줘",
            "📊 데이터의 상관관계를 히트맵으로 보여줘",
            "🔍 결측값을 자동으로 처리해줘",
            "🤖 가장 적합한 ML 알고리즘을 추천해줘"
        ]
        
        # 데이터 특성에 따른 맞춤형 질문 추가
        if potential_targets:
            followups.append(f"🎯 '{potential_targets[0]}' 컬럼을 타겟으로 분류 모델을 만들어줘")
        if len(numeric_cols) >= 3:
            followups.append("📈 PCA로 차원 축소하고 시각화해줘")
        if shape[0] > 10000:
            followups.append("🧠 딥러닝 모델을 만들어줘")
        if len(categorical_cols) > 0:
            followups.append("🏷️ 범주형 데이터 분포를 시각화해줘")
        
        # 분석 결과를 세션 상태에 저장
        analysis_result = {
            "content": greeting_content,
            "visualizations": visualizations,
            "followups": followups[:6]  # 최대 6개로 제한
        }
        
        # 분석 결과를 별도 저장 (초기화 완료 후 사용)
        st.session_state.auto_analysis_result = analysis_result
        
        return True
        
    except Exception as e:
        logging.error(f"고급 자동 데이터 분석 중 오류: {e}")
        # 간단한 인사말만 저장
        simple_greeting = f"""🎉 **AI Data Scientist 환경에 오신 것을 환영합니다!**

📊 **데이터가 성공적으로 로드되었습니다**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열

🚀 **빠른 시작:**
- `auto_ml_pipeline(df, 'target_col')` - 완전 자동 ML
- `df.describe()` - 기초 통계
- `df.hist(); plt.show()` - 분포 시각화

무엇을 분석하고 싶으신지 말씀해 주세요! 🤖✨"""
        
        st.session_state.auto_analysis_result = {
            "content": simple_greeting,
            "visualizations": [],
            "followups": [
                "🚀 auto_ml_pipeline(df) 실행해줘", 
                "📊 데이터의 기본 정보를 보여줘", 
                "📈 전체 변수 히스토그램을 그려줘",
                "🔍 결측값 현황을 확인해줘"
            ]
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
                        
                        # 🆕 MCP tools 준비
                        extra_tools = [enhanced_python_tool]
                        if tools:
                            extra_tools.extend(tools)
                            logging.debug(f"Added {len(tools)} MCP tools to pandas agent")
                        
                        # 🆕 pandas agent를 enhanced tools와 함께 생성
                        pandas_agent = create_pandas_dataframe_agent(
                            model,
                            df,
                            verbose=True,
                            agent_type=AgentType.OPENAI_FUNCTIONS,
                            allow_dangerous_code=True,
                            prefix=st.session_state.selected_prompt_text,
                            handle_parsing_errors=True,
                            max_iterations=10,
                            early_stopping_method="generate",
                            extra_tools=extra_tools  # 🔧 enhanced tool을 여기서 포함
                        )
                        
                        logging.debug(f"Enhanced pandas agent created with {len(extra_tools)} extra tools")
                        
                        # Ensure the agent supports streaming
                        if hasattr(pandas_agent, 'llm'):
                            pandas_agent.llm.streaming = True
                        elif hasattr(pandas_agent, 'agent') and hasattr(pandas_agent.agent, 'llm_chain') and hasattr(pandas_agent.agent.llm_chain, 'llm'):
                            pandas_agent.agent.llm_chain.llm.streaming = True
                            
                        st.session_state.agent = pandas_agent
                        st.session_state.agent_type = "pandas"
                        logging.debug('Enhanced pandas agent with auto_ml_pipeline functions created successfully')
                        
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
                            
                            **🤖 AutoML 특별 기능:**
                            - auto_ml_pipeline() - 완전 자동화 ML
                            - auto_detect_problem_type() - 문제 유형 자동 감지
                            - auto_select_models() - 최적 모델 자동 선택
                            - generate_ml_report() - 분석 보고서 생성
                            
                            **과학계산:**
                            - scipy
                            
                            **추천 시작 명령어:**
                            - `auto_ml_pipeline(df, 'target_col')` - AutoML 실행
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
                    
                    # 🆕 강화된 오류 처리 및 복구 시스템
                    final_output = ""
                    error_occurred = False
                    error_messages = []
                    partial_results = []
                    step_count = 0
                    successful_steps = 0
                    
                    try:
                        # Use agent.stream() for pandas agent
                        response_stream = st.session_state.agent.stream({"input": query})
                        
                        # Process each step in the stream
                        for step in response_stream:
                            step_count += 1
                            logging.debug(f"Pandas agent step {step_count}: {type(step)} - {list(step.keys()) if isinstance(step, dict) else step}")
                            
                            try:
                                # Process the step using our callback
                                streaming_callback({"node": "pandas_agent", "content": step})
                                successful_steps += 1
                                
                                # Extract and store partial results
                                if isinstance(step, dict):
                                    if "output" in step:
                                        final_output = step["output"]
                                    elif "intermediate_steps" in step:
                                        # 중간 단계 결과 저장
                                        intermediate = step["intermediate_steps"]
                                        if intermediate:
                                            partial_results.append(f"Step {step_count}: {str(intermediate)[:200]}...")
                                    elif "action" in step:
                                        # 실행된 액션 정보 저장
                                        action = step["action"]
                                        partial_results.append(f"Action: {action.get('tool', 'Unknown')} - {action.get('tool_input', {})}")
                                
                                # 🆕 텍스트 응답이 있으면 실시간으로 업데이트
                                if isinstance(step, str) and step.strip():
                                    if not final_output:
                                        final_output = step
                                    else:
                                        final_output += "\n" + step
                                    # 실시간으로 텍스트 표시 업데이트
                                    text_placeholder.markdown(final_output)
                                        
                            except Exception as step_error:
                                # 🆕 상세한 오류 정보 수집
                                error_occurred = True
                                error_detail = {
                                    'step': step_count,
                                    'error': str(step_error),
                                    'error_type': type(step_error).__name__,
                                    'step_data': str(step)[:100] + "..." if len(str(step)) > 100 else str(step)
                                }
                                error_messages.append(error_detail)
                                
                                logging.warning(f"Error in pandas agent step {step_count}: {step_error}")
                                
                                # 🆕 오류 정보를 사용자에게 친화적으로 표시
                                friendly_error = f"⚠️ 단계 {step_count}에서 오류 발생 (계속 진행 중...)"
                                tool_placeholder.markdown(f"\n**{friendly_error}**\n")
                                
                                # 🆕 오류 유형별 대응
                                if "DataFrame" in str(step_error) and "empty" in str(step_error):
                                    partial_results.append(f"Step {step_count}: 빈 DataFrame 오류 - 데이터 필터링 결과가 비어있을 수 있습니다.")
                                elif "KeyError" in str(step_error):
                                    partial_results.append(f"Step {step_count}: 컬럼명 오류 - 존재하지 않는 컬럼을 참조했을 수 있습니다.")
                                elif "ValueError" in str(step_error):
                                    partial_results.append(f"Step {step_count}: 값 오류 - 데이터 타입이나 형식 문제가 있을 수 있습니다.")
                                else:
                                    partial_results.append(f"Step {step_count}: {error_detail['error_type']} 오류")
                                
                                # 계속 진행
                                continue
                        
                    except Exception as stream_error:
                        # 🆕 스트림 자체에서 오류가 발생한 경우
                        error_occurred = True
                        error_detail = {
                            'step': 'stream_initialization',
                            'error': str(stream_error),
                            'error_type': type(stream_error).__name__,
                            'step_data': 'stream_creation'
                        }
                        error_messages.append(error_detail)
                        
                        logging.error(f"Pandas agent stream error: {stream_error}")
                        
                        # 🆕 스트림 오류 시 기본 정보 제공 시도
                        try:
                            if st.session_state.dataframe is not None:
                                df = st.session_state.dataframe
                                fallback_info = f"""❌ 데이터 분석 중 오류가 발생했습니다.

**오류 내용:** {str(stream_error)}

🔍 **현재 데이터 상태:**
- 데이터 크기: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- 컬럼: {', '.join(df.columns.tolist()[:5])}{'...' if len(df.columns) > 5 else ''}

💡 **추천 해결 방법:**
1. 더 간단한 질문으로 시작해보세요: "데이터의 첫 5행을 보여줘"
2. 구체적인 컬럼명을 사용해보세요: "'{df.columns[0]}' 컬럼의 통계를 보여줘"
3. 단계별로 나누어 질문해보세요: "데이터 정보부터 알려줘"
"""
                                final_output = fallback_info
                                text_placeholder.markdown(final_output)
                            else:
                                final_output = f"❌ 데이터 분석 오류: {str(stream_error)}\n\n데이터를 다시 로드해주세요."
                                text_placeholder.markdown(final_output)
                        except Exception as fallback_error:
                            logging.error(f"Fallback info generation error: {fallback_error}")
                            final_output = "❌ 분석 중 오류가 발생했습니다. 데이터를 다시 확인해주세요."
                            text_placeholder.markdown(final_output)
                    
                    # 🆕 결과 종합 및 사용자 친화적 응답 생성
                    if error_occurred:
                        # 성공한 단계와 실패한 단계 비율 계산
                        success_rate = (successful_steps / max(step_count, 1)) * 100 if step_count > 0 else 0
                        
                        # 🆕 부분적 성공 시 결과 보강
                        if final_output and success_rate > 50:
                            # 대부분 성공한 경우
                            final_output += f"\n\n⚠️ **처리 완료** (성공률: {success_rate:.1f}%)\n"
                            final_output += f"📊 총 {step_count}단계 중 {successful_steps}단계 성공\n"
                            
                            if len(error_messages) <= 2:
                                final_output += f"🔧 일부 오류가 있었지만 주요 결과는 정상 처리되었습니다."
                            else:
                                final_output += f"🔧 {len(error_messages)}개의 오류가 있었지만 결과를 생성했습니다."
                        
                        # 🆕 부분적 실패 시 대안 제시
                        elif final_output and success_rate <= 50:
                            final_output += f"\n\n⚠️ **부분 처리 완료** (성공률: {success_rate:.1f}%)\n"
                            final_output += "🔄 더 정확한 결과를 위해 다음 중 하나를 시도해보세요:\n"
                            final_output += "1. 더 구체적인 질문으로 재시도\n"
                            final_output += "2. 단계별로 나누어 질문\n"
                            final_output += "3. 다른 접근 방법 사용\n"
                            
                            # 부분 결과가 있으면 표시
                            if partial_results:
                                final_output += f"\n📋 **처리된 단계들:**\n"
                                for result in partial_results[-3:]:  # 최근 3개만 표시
                                    final_output += f"- {result}\n"
                        
                        # 🆕 완전 실패 시 복구 시도
                        elif not final_output:
                            final_output = f"❌ **분석 중 오류 발생** ({len(error_messages)}개 오류)\n\n"
                            
                            # 🆕 데이터 기반 대안 제시
                            try:
                                if st.session_state.dataframe is not None:
                                    df = st.session_state.dataframe
                                    
                                    # 🆕 자동 복구 시도 - 기본 정보 제공
                                    auto_recovery = f"""🔄 **자동 복구 시도 결과:**

📊 **데이터 기본 정보:**
- 크기: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- 컬럼: {', '.join(df.columns.tolist()[:5])}{'...' if len(df.columns) > 5 else ''}
- 수치형 컬럼: {len(df.select_dtypes(include=['number']).columns)}개
- 범주형 컬럼: {len(df.select_dtypes(include=['object', 'category']).columns)}개

💡 **다음을 시도해보세요:**
1. `df.head()` - 데이터 미리보기
2. `df.info()` - 컬럼 정보 확인  
3. `df.describe()` - 기초 통계
4. `df.columns.tolist()` - 정확한 컬럼명 확인

🎯 **추천 질문 예시:**
- "데이터의 기본 정보를 알려줘"
- "첫 10행을 보여줘"
- "수치형 컬럼들의 통계를 보여줘"
"""
                                    final_output += auto_recovery
                                else:
                                    final_output += "📂 데이터가 로드되지 않았습니다. CSV 파일을 먼저 업로드해주세요."
                                    
                            except Exception as recovery_error:
                                logging.error(f"Auto recovery failed: {recovery_error}")
                                final_output += "🔄 자동 복구에 실패했습니다. 데이터를 다시 로드해주세요."
                            
                            # 🆕 오류 요약 정보 추가
                            if len(error_messages) <= 3:
                                final_output += f"\n\n🔍 **오류 상세:**\n"
                                for i, err in enumerate(error_messages, 1):
                                    final_output += f"{i}. {err['error_type']}: {err['error'][:100]}{'...' if len(err['error']) > 100 else ''}\n"
                    
                    # 🆕 최종 텍스트가 없으면 기본 응답
                    if not final_output:
                        final_output = "🤔 응답을 생성하지 못했습니다. 다른 방식으로 질문해보세요."
                    
                    # 🆕 최종 응답 표시
                    if not accumulated_text_obj or accumulated_text_obj == [""]:
                        accumulated_text_obj.append(final_output)
                        text_placeholder.markdown("".join(accumulated_text_obj))
                    
                    response = {"output": final_output}
                    
                    # 🆕 디버깅 정보 로그
                    logging.info(f"Pandas agent processing completed - Steps: {step_count}, Successful: {successful_steps}, Errors: {len(error_messages)}")
                
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