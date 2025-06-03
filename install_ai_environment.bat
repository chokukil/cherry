@echo off
REM AI Data Scientist 환경 설정 스크립트 (Windows Batch)
REM 파일명: install_ai_environment.bat

setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1

REM 색상 코드 설정
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "CYAN=[96m"
set "WHITE=[97m"
set "RESET=[0m"

REM 헤더 출력
cls
echo %GREEN%🚀 AI Data Scientist 환경 설정을 시작합니다...%RESET%
echo %YELLOW%============================================================%RESET%
echo %CYAN%운영체제: %OS%%RESET%
echo %CYAN%현재 경로: %CD%%RESET%
echo %YELLOW%============================================================%RESET%
echo.

REM 관리자 권한 확인
net session >nul 2>&1
if %errorLevel% == 0 (
    echo %GREEN%✅ 관리자 권한으로 실행 중%RESET%
) else (
    echo %YELLOW%⚠️ 일반 사용자 권한으로 실행 중 (일부 기능 제한될 수 있음)%RESET%
)
echo.

REM GPU 감지 함수
:detect_gpu
echo %YELLOW%🔍 GPU 환경 감지 중...%RESET%
set "has_nvidia=false"
set "has_amd=false"

REM NVIDIA GPU 확인
nvidia-smi --list-gpus >nul 2>&1
if !errorlevel! == 0 (
    set "has_nvidia=true"
    echo %GREEN%✅ NVIDIA GPU 감지됨:%RESET%
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>nul | findstr /r ".*"
) else (
    echo %RED%❌ NVIDIA GPU 감지되지 않음%RESET%
)

REM WMI를 통한 GPU 정보 확인
for /f "skip=1 delims=" %%i in ('wmic path win32_videocontroller get name /format:list 2^>nul ^| findstr "Name="') do (
    set "gpu_name=%%i"
    set "gpu_name=!gpu_name:Name=!"
    echo !gpu_name! | findstr /i "nvidia geforce quadro tesla" >nul 2>&1
    if !errorlevel! == 0 (
        set "has_nvidia=true"
    )
    echo !gpu_name! | findstr /i "amd radeon" >nul 2>&1
    if !errorlevel! == 0 (
        set "has_amd=true"
        echo %YELLOW%⚠️ AMD GPU 감지됨 (CUDA 지원 안됨): !gpu_name!%RESET%
    )
)

echo.
echo %CYAN%📊 감지된 GPU 정보:%RESET%
if "!has_nvidia!"=="true" (
    echo %WHITE%  - NVIDIA GPU: ✅%RESET%
) else (
    echo %WHITE%  - NVIDIA GPU: ❌%RESET%
)
if "!has_amd!"=="true" (
    echo %WHITE%  - AMD GPU: ✅%RESET%
) else (
    echo %WHITE%  - AMD GPU: ❌%RESET%
)
echo.
goto :eof

REM uv 설치 확인
:check_uv
echo %YELLOW%🔍 uv 설치 확인 중...%RESET%
uv --version >nul 2>&1
if !errorlevel! == 0 (
    for /f "delims=" %%i in ('uv --version 2^>nul') do set "uv_version=%%i"
    echo %GREEN%✅ uv 발견: !uv_version!%RESET%
    goto :eof
) else (
    echo %RED%❌ uv가 설치되지 않았습니다.%RESET%
    echo %YELLOW%자동으로 uv를 설치하시겠습니까? (y/n): %RESET%
    set /p "install_uv="
    if /i "!install_uv!"=="y" (
        call :install_uv
    ) else if /i "!install_uv!"=="" (
        call :install_uv
    ) else (
        echo %RED%uv 설치가 필요합니다. 스크립트를 종료합니다.%RESET%
        pause
        exit /b 1
    )
)
goto :eof

REM uv 설치
:install_uv
echo %GREEN%📦 uv 설치 중...%RESET%
REM PowerShell을 사용한 uv 설치
powershell -Command "& {Invoke-WebRequest -Uri 'https://astral.sh/uv/install.ps1' -OutFile 'install_uv.ps1'; PowerShell -ExecutionPolicy Bypass -File 'install_uv.ps1'; Remove-Item 'install_uv.ps1' -Force}"
if !errorlevel! == 0 (
    REM 환경변수 새로고침
    call :refresh_env
    uv --version >nul 2>&1
    if !errorlevel! == 0 (
        echo %GREEN%✅ uv 설치 완료!%RESET%
    ) else (
        echo %RED%❌ uv 설치 실패. 수동으로 설치해주세요.%RESET%
        echo %YELLOW%설치 방법: https://docs.astral.sh/uv/getting-started/installation/%RESET%
        pause
        exit /b 1
    )
) else (
    echo %RED%❌ uv 설치 중 오류 발생%RESET%
    echo %YELLOW%수동 설치 후 다시 실행해주세요.%RESET%
    pause
    exit /b 1
)
goto :eof

REM 환경변수 새로고침
:refresh_env
for /f "usebackq tokens=2,*" %%A in (`reg query HKCU\Environment /v PATH 2^>nul`) do set "UserPath=%%B"
for /f "usebackq tokens=2,*" %%A in (`reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH 2^>nul`) do set "SystemPath=%%B"
set "PATH=%SystemPath%;%UserPath%"
goto :eof

REM 모드 선택
:select_mode
echo %YELLOW%📋 설치 옵션을 선택하세요:%RESET%
echo %WHITE%1) 🖥️  CPU 전용 (안정적, 모든 환경)%RESET%
echo %WHITE%2) 🚀 GPU 가속 (NVIDIA GPU 필요)%RESET%
echo %WHITE%3) 🔧 자동 감지 (추천)%RESET%
echo %WHITE%4) 🧪 연구용 전체 패키지%RESET%
echo.
set /p "choice=선택하세요 (1-4, 기본값: 3): "
if "!choice!"=="" set "choice=3"

if "!choice!"=="1" (
    set "install_mode=cpu"
    echo %YELLOW%🖥️ CPU 전용 설치를 시작합니다...%RESET%
) else if "!choice!"=="2" (
    set "install_mode=gpu"
    echo %YELLOW%🚀 GPU 가속 설치를 시작합니다...%RESET%
) else if "!choice!"=="3" (
    if "!has_nvidia!"=="true" (
        echo %YELLOW%GPU가 감지되었습니다. GPU 버전을 설치하시겠습니까? (y/n, 기본값: y): %RESET%
        set /p "gpu_choice="
        if /i "!gpu_choice!"=="n" (
            set "install_mode=cpu"
        ) else (
            set "install_mode=gpu"
        )
    ) else (
        echo %YELLOW%GPU가 감지되지 않았습니다. CPU 모드로 설치합니다.%RESET%
        set "install_mode=cpu"
    )
) else if "!choice!"=="4" (
    set "install_mode=full"
    echo %YELLOW%🧪 연구용 전체 패키지 설치를 시작합니다...%RESET%
) else (
    echo %YELLOW%⚠️ 잘못된 선택입니다. 자동 감지로 설치합니다...%RESET%
    goto select_mode
)
echo.
goto :eof

REM uv 명령 실행 헬퍼
:run_uv_command
set "command=%~1"
echo %YELLOW%🔄 실행 중: %command%%RESET%
%command%
if !errorlevel! == 0 (
    echo %GREEN%✅ 성공: %command%%RESET%
    set "cmd_success=true"
) else (
    echo %RED%❌ 실패: %command% (Exit Code: !errorlevel!)%RESET%
    set "cmd_success=false"
)
goto :eof

REM 기본 패키지 설치
:install_base
echo %GREEN%📦 기본 패키지 설치 중...%RESET%
call :run_uv_command "uv sync"
echo.
goto :eof

REM CPU 버전 설치
:install_cpu
echo %GREEN%🖥️ CPU 전용 ML/DL 패키지 설치 중...%RESET%

set cpu_packages=tensorflow-cpu>=2.15.0 torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 xgboost>=2.0.0 lightgbm>=4.0.0 catboost>=1.2.0 shap>=0.45.0 statsmodels>=0.14.0 plotly>=5.15.0 wordcloud>=1.9.0 openpyxl>=3.1.0 nltk>=3.8.0 Pillow>=10.0.0

for %%p in (%cpu_packages%) do (
    call :run_uv_command "uv add ""%%p"""
    timeout /t 1 /nobreak >nul
)
echo.
goto :eof

REM GPU 버전 설치
:install_gpu
echo %GREEN%🚀 GPU 가속 ML/DL 패키지 설치 중...%RESET%

set gpu_packages=tensorflow[and-cuda]>=2.15.0 torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 xgboost>=2.0.0 lightgbm>=4.0.0 catboost>=1.2.0 shap>=0.45.0 statsmodels>=0.14.0 transformers>=4.30.0 datasets>=2.12.0 accelerate>=0.20.0 plotly>=5.15.0 wordcloud>=1.9.0 openpyxl>=3.1.0 nltk>=3.8.0 Pillow>=10.0.0

for %%p in (%gpu_packages%) do (
    call :run_uv_command "uv add ""%%p"""
    timeout /t 1 /nobreak >nul
)
echo.
goto :eof

REM 전체 버전 설치
:install_full
echo %GREEN%🎯 전체 패키지 설치 중...%RESET%

REM GPU 패키지 먼저
call :install_gpu

REM 추가 연구/MLOps 패키지
set additional_packages=optuna>=3.2.0 mlflow>=2.5.0 wandb>=0.15.0 bokeh>=3.1.0 altair>=5.0.0 holoviews>=1.17.0

echo %GREEN%📊 추가 연구용 패키지 설치 중...%RESET%
for %%p in (%additional_packages%) do (
    call :run_uv_command "uv add ""%%p"""
    timeout /t 1 /nobreak >nul
)
echo.
goto :eof

REM 설정 파일 생성
:create_config
echo %GREEN%📄 설정 파일 생성 중...%RESET%

(
echo """
echo AI Data Scientist 환경 설정 (Windows Batch^)
echo 설치 모드: %install_mode%
echo 설치 시간: %date% %time%
echo OS: %OS%
echo """
echo.
echo import platform
echo import sys
echo.
echo # 환경 설정
echo INSTALL_MODE = "%install_mode%"
echo CUDA_AVAILABLE = False
echo TENSORRT_AVAILABLE = False
echo OS_TYPE = "Windows"
echo.
echo def get_environment_info^(^):
echo     """환경 정보를 반환합니다."""
echo     info = {
echo         'install_mode': INSTALL_MODE,
echo         'os': platform.system^(^),
echo         'python_version': sys.version,
echo         'packages': [],
echo         'gpu_support': False
echo     }
echo     
echo     # TensorFlow 확인
echo     try:
echo         import tensorflow as tf
echo         info['packages'].append^(f'tensorflow=={tf.__version__}'^)
echo         gpu_devices = tf.config.list_physical_devices^('GPU'^)
echo         if gpu_devices:
echo             info['gpu_support'] = True
echo             global CUDA_AVAILABLE
echo             CUDA_AVAILABLE = True
echo             info['gpu_devices'] = [device.name for device in gpu_devices]
echo     except ImportError:
echo         pass
echo     
echo     # PyTorch 확인
echo     try:
echo         import torch
echo         info['packages'].append^(f'torch=={torch.__version__}'^)
echo         if torch.cuda.is_available^(^):
echo             info['gpu_support'] = True
echo             info['torch_cuda_devices'] = torch.cuda.device_count^(^)
echo     except ImportError:
echo         pass
echo     
echo     # 기타 패키지들 확인
echo     packages_to_check = ['xgboost', 'lightgbm', 'catboost', 'shap', 'sklearn', 'pandas', 'numpy']
echo     for pkg in packages_to_check:
echo         try:
echo             module = __import__^(pkg^)
echo             if hasattr^(module, '__version__'^):
echo                 info['packages'].append^(f'{pkg}=={module.__version__}'^)
echo             else:
echo                 info['packages'].append^(f'{pkg}=installed'^)
echo         except ImportError:
echo             pass
echo     
echo     return info
echo.
echo def print_system_info^(^):
echo     """시스템 정보를 출력합니다."""
echo     info = get_environment_info^(^)
echo     print^("🖥️ AI Data Scientist 환경 정보 (Windows Batch^):"^)
echo     print^(f"   OS: {info['os']}'^)
echo     print^(f"   Python: {info['python_version']}'^)
echo     print^(f"   설치 모드: {info['install_mode']}'^)
echo     print^(f"   GPU 지원: {'✅' if info['gpu_support'] else '❌'}'^)
echo     print^(f"   설치된 패키지: {len^(info['packages'^)^)}개"^)
echo     for pkg in info['packages']:
echo         print^(f"     - {pkg}"^)
echo.
echo if __name__ == "__main__":
echo     print_system_info^(^)
) > ai_config.py

echo %GREEN%✅ ai_config.py 설정 파일이 생성되었습니다.%RESET%
echo.
goto :eof

REM GPU 유틸리티 생성
:create_gpu_utils
echo %GREEN%📄 GPU 유틸리티 모듈 생성 중...%RESET%

(
echo """
echo Windows Batch용 GPU 감지 및 최적화 모듈
echo """
echo.
echo def check_gpu_availability^(^):
echo     """Windows 환경에서 GPU 가용성을 확인합니다."""
echo     import subprocess
echo     import sys
echo     
echo     gpu_status = {
echo         'nvidia_smi': False,
echo         'tensorflow_gpu': False,
echo         'torch_gpu': False,
echo         'gpu_devices': [],
echo         'recommendations': []
echo     }
echo     
echo     # NVIDIA-SMI 확인
echo     try:
echo         result = subprocess.run^(['nvidia-smi.exe', '--list-gpus'], capture_output=True, text=True, timeout=10^)
echo         if result.returncode == 0:
echo             gpu_status['nvidia_smi'] = True
echo             gpu_status['gpu_devices'].extend^(result.stdout.strip^(^).split^('\n'^)^)
echo             print^("✅ NVIDIA-SMI 사용 가능"^)
echo         else:
echo             print^("❌ NVIDIA-SMI 사용 불가"^)
echo     except Exception as e:
echo         print^(f"❌ NVIDIA-SMI 체크 실패: {e}"^)
echo     
echo     # TensorFlow GPU 확인
echo     try:
echo         import tensorflow as tf
echo         gpu_devices = tf.config.list_physical_devices^('GPU'^)
echo         if gpu_devices:
echo             gpu_status['tensorflow_gpu'] = True
echo             print^(f"✅ TensorFlow GPU 사용 가능: {len^(gpu_devices^)}개 디바이스"^)
echo         else:
echo             print^("⚠️ TensorFlow GPU 사용 불가"^)
echo             gpu_status['recommendations'].append^("GPU 버전 TensorFlow 설치 고려"^)
echo     except ImportError:
echo         print^("❌ TensorFlow 설치되지 않음"^)
echo         gpu_status['recommendations'].append^("TensorFlow 설치 필요"^)
echo     
echo     # PyTorch GPU 확인
echo     try:
echo         import torch
echo         if torch.cuda.is_available^(^):
echo             gpu_status['torch_gpu'] = True
echo             gpu_count = torch.cuda.device_count^(^)
echo             print^(f"✅ PyTorch CUDA 사용 가능: {gpu_count}개 디바이스"^)
echo         else:
echo             print^("⚠️ PyTorch CUDA 사용 불가"^)
echo             gpu_status['recommendations'].append^("CUDA 지원 PyTorch 설치 고려"^)
echo     except ImportError:
echo         print^("❌ PyTorch 설치되지 않음"^)
echo     
echo     return gpu_status
echo.
echo if __name__ == "__main__":
echo     print^("🔍 Windows GPU 환경 확인 중..."^)
echo     status = check_gpu_availability^(^)
echo     
echo     if status['recommendations']:
echo         print^("\n💡 권장사항:"^)
echo         for rec in status['recommendations']:
echo             print^(f"  - {rec}"^)
) > gpu_utils.py

echo %GREEN%✅ gpu_utils.py 모듈 생성 완료%RESET%
echo.
goto :eof

REM 설치 검증
:verify_installation
echo %YELLOW%🔍 설치 검증 중...%RESET%

python -c "
import sys
packages = ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn']
missing = []

print('📋 패키지 설치 확인:')
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {pkg}=={version}')
    except ImportError:
        print(f'❌ {pkg}')
        missing.append(pkg)

# TensorFlow 확인
try:
    import tensorflow as tf
    print(f'✅ tensorflow=={tf.__version__}')
    gpu_count = len(tf.config.list_physical_devices('GPU'))
    print(f'   GPU 디바이스: {gpu_count}개')
except ImportError:
    print('❌ tensorflow')
    missing.append('tensorflow')

# PyTorch 확인
try:
    import torch
    print(f'✅ torch=={torch.__version__}')
    if torch.cuda.is_available():
        print(f'   CUDA 사용 가능: {torch.cuda.device_count()}개 디바이스')
    else:
        print('   CUDA 사용 불가')
except ImportError:
    print('❌ torch')
    missing.append('torch')

# ML 패키지 확인
ml_packages = ['xgboost', 'lightgbm', 'catboost', 'shap']
for pkg in ml_packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {pkg}=={version}')
    except ImportError:
        print(f'❌ {pkg}')
        missing.append(pkg)

if missing:
    print(f'\n⚠️ 누락된 패키지: {missing}')
    sys.exit(1)
else:
    print('\n🎉 모든 패키지가 정상적으로 설치되었습니다!')
"

if !errorlevel! == 0 (
    set "verification_success=true"
    echo %GREEN%✅ 설치 검증 완료!%RESET%
) else (
    set "verification_success=false"
    echo %RED%❌ 설치 검증 실패!%RESET%
)
echo.
goto :eof

REM 사용법 안내
:show_usage
echo %GREEN%🎯 사용법 안내:%RESET%
echo %WHITE%1. Streamlit 앱 시작:%RESET%
echo %CYAN%   streamlit run Home.py%RESET%
echo.
echo %WHITE%2. 환경 정보 확인:%RESET%
echo %CYAN%   python ai_config.py%RESET%
echo.
echo %WHITE%3. GPU 테스트:%RESET%
echo %CYAN%   python gpu_utils.py%RESET%
echo.
echo %WHITE%4. 패키지 업데이트:%RESET%
echo %CYAN%   uv sync%RESET%
echo.
echo %WHITE%5. PowerShell 실행 정책 문제 시:%RESET%
echo %CYAN%   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser%RESET%
echo.
goto :eof

REM 메인 실행
:main
REM 1. GPU 감지
call :detect_gpu

REM 2. uv 확인
call :check_uv

REM 3. 모드 선택
call :select_mode

REM 4. 기본 패키지 설치
echo %GREEN%📦 3단계: 기본 패키지 설치%RESET%
call :install_base

REM 5. 모드별 설치
echo %GREEN%🧠 4단계: ML/DL 패키지 설치%RESET%
if "!install_mode!"=="cpu" (
    call :install_cpu
) else if "!install_mode!"=="gpu" (
    call :install_gpu
) else if "!install_mode!"=="full" (
    call :install_full
) else (
    echo %RED%❌ 알 수 없는 설치 모드: !install_mode!%RESET%
    goto end
)

REM 6. 설정 파일 생성
echo %GREEN%🛠️ 5단계: 설정 파일 생성%RESET%
call :create_config
call :create_gpu_utils

REM 7. 설치 검증
echo %GREEN%🔍 6단계: 설치 검증%RESET%
call :verify_installation

REM 8. 결과 출력
echo %YELLOW%============================================================%RESET%
if "!verification_success!"=="true" (
    echo %GREEN%🎉 AI Data Scientist 환경 설정이 완료되었습니다!%RESET%
    echo.
    call :show_usage
) else (
    echo %RED%❌ 설치 중 일부 오류가 발생했습니다.%RESET%
    echo %YELLOW%로그를 확인하고 수동으로 누락된 패키지를 설치해주세요.%RESET%
)

:end
echo.
echo %YELLOW%아무 키나 눌러서 종료하세요...%RESET%
pause >nul
exit /b