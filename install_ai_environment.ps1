# AI Data Scientist 환경 설정 스크립트 (Windows PowerShell)
# 파일명: install_ai_environment.ps1

param(
    [string]$Mode = "auto",
    [switch]$Force,
    [switch]$Help
)

# 도움말 함수
function Show-Help {
    Write-Host @"
🚀 AI Data Scientist 환경 설정 스크립트

사용법:
  .\install_ai_environment.ps1 [옵션]

옵션:
  -Mode <모드>    설치 모드 선택
                  - auto: 자동 감지 (기본값)
                  - cpu: CPU 전용
                  - gpu: GPU 가속
                  - full: 전체 패키지
  -Force          기존 패키지 강제 재설치
  -Help           이 도움말 표시

예시:
  .\install_ai_environment.ps1                    # 자동 감지
  .\install_ai_environment.ps1 -Mode cpu          # CPU 전용
  .\install_ai_environment.ps1 -Mode gpu -Force   # GPU 강제 설치

"@ -ForegroundColor Green
}

# 스크립트 헤더
function Write-Header {
    Clear-Host
    Write-Host "🚀 AI Data Scientist 환경 설정을 시작합니다..." -ForegroundColor Green
    Write-Host "=" * 60 -ForegroundColor Yellow
    Write-Host "운영체제: $($env:OS)" -ForegroundColor Cyan
    Write-Host "PowerShell 버전: $($PSVersionTable.PSVersion)" -ForegroundColor Cyan
    Write-Host "현재 경로: $(Get-Location)" -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Yellow
}

# GPU 감지 함수
function Test-GPU {
    Write-Host "🔍 GPU 환경 감지 중..." -ForegroundColor Yellow
    
    $gpuInfo = @{
        HasNvidia = $false
        HasAMD = $false
        CudaAvailable = $false
        GPUNames = @()
    }
    
    try {
        # NVIDIA GPU 확인 (nvidia-smi)
        $nvidiaSmi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
        if ($nvidiaSmi) {
            $result = & nvidia-smi.exe --list-gpus 2>$null
            if ($LASTEXITCODE -eq 0) {
                $gpuInfo.HasNvidia = $true
                $gpuInfo.GPUNames = $result
                Write-Host "✅ NVIDIA GPU 감지됨:" -ForegroundColor Green
                $result | ForEach-Object { Write-Host "   $_" -ForegroundColor White }
            }
        }
        
        # WMI를 통한 GPU 정보 확인
        $gpuDevices = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -notlike "*Basic*" }
        foreach ($gpu in $gpuDevices) {
            if ($gpu.Name -match "NVIDIA|GeForce|Quadro|Tesla") {
                $gpuInfo.HasNvidia = $true
                if ($gpu.Name -notin $gpuInfo.GPUNames) {
                    $gpuInfo.GPUNames += $gpu.Name
                }
            }
            elseif ($gpu.Name -match "AMD|Radeon") {
                $gpuInfo.HasAMD = $true
                Write-Host "⚠️ AMD GPU 감지됨 (CUDA 지원 안됨): $($gpu.Name)" -ForegroundColor Yellow
            }
        }
        
        # CUDA 테스트 (Python이 설치된 경우)
        try {
            $pythonPath = Get-Command "python.exe" -ErrorAction SilentlyContinue
            if ($pythonPath) {
                $cudaTest = python -c "
import subprocess
import sys
try:
    import tensorflow as tf
    print('CUDA_AVAILABLE:' + str(len(tf.config.list_physical_devices('GPU')) > 0))
except:
    print('CUDA_AVAILABLE:False')
" 2>$null
                if ($cudaTest -match "CUDA_AVAILABLE:True") {
                    $gpuInfo.CudaAvailable = $true
                }
            }
        }
        catch {
            # Python이 없거나 오류 발생 시 무시
        }
        
    }
    catch {
        Write-Host "⚠️ GPU 감지 중 오류 발생: $($_.Exception.Message)" -ForegroundColor Yellow
    }
    
    # 결과 출력
    Write-Host "`n📊 감지된 GPU 정보:" -ForegroundColor Cyan
    Write-Host "  - NVIDIA GPU: $(if($gpuInfo.HasNvidia){'✅'}else{'❌'})" -ForegroundColor White
    Write-Host "  - AMD GPU: $(if($gpuInfo.HasAMD){'✅'}else{'❌'})" -ForegroundColor White
    Write-Host "  - CUDA 사용 가능: $(if($gpuInfo.CudaAvailable){'✅'}else{'❌'})" -ForegroundColor White
    
    return $gpuInfo
}

# uv 설치 확인 및 설치
function Test-UV {
    Write-Host "🔍 uv 설치 확인 중..." -ForegroundColor Yellow
    
    $uvPath = Get-Command "uv.exe" -ErrorAction SilentlyContinue
    if (-not $uvPath) {
        Write-Host "❌ uv가 설치되지 않았습니다." -ForegroundColor Red
        Write-Host "자동으로 uv를 설치하시겠습니까? (y/n)" -ForegroundColor Yellow -NoNewline
        $response = Read-Host
        
        if ($response -eq 'y' -or $response -eq 'Y' -or $response -eq '') {
            Write-Host "📦 uv 설치 중..." -ForegroundColor Green
            try {
                # uv 설치 (Windows)
                Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile "install_uv.ps1"
                PowerShell -ExecutionPolicy Bypass -File "install_uv.ps1"
                Remove-Item "install_uv.ps1" -Force
                
                # 환경 변수 새로고침
                $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
                
                # 다시 확인
                $uvPath = Get-Command "uv.exe" -ErrorAction SilentlyContinue
                if ($uvPath) {
                    Write-Host "✅ uv 설치 완료!" -ForegroundColor Green
                } else {
                    Write-Host "❌ uv 설치 실패. 수동으로 설치해주세요." -ForegroundColor Red
                    Write-Host "설치 방법: https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Yellow
                    exit 1
                }
            }
            catch {
                Write-Host "❌ uv 설치 중 오류 발생: $($_.Exception.Message)" -ForegroundColor Red
                Write-Host "수동 설치 후 다시 실행해주세요." -ForegroundColor Yellow
                exit 1
            }
        } else {
            Write-Host "uv 설치가 필요합니다. 스크립트를 종료합니다." -ForegroundColor Red
            exit 1
        }
    } else {
        $uvVersion = & uv.exe --version 2>$null
        Write-Host "✅ uv 발견: $uvVersion" -ForegroundColor Green
    }
}

# uv 명령 실행
function Invoke-UVCommand {
    param([string]$Command)
    
    Write-Host "🔄 실행 중: $Command" -ForegroundColor Yellow
    try {
        $result = Invoke-Expression $Command
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ 성공: $Command" -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ 실패: $Command (Exit Code: $LASTEXITCODE)" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "❌ 예외 발생: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# 기본 패키지 설치
function Install-BasePackages {
    Write-Host "`n📦 기본 패키지 설치 중..." -ForegroundColor Green
    
    $success = Invoke-UVCommand "uv sync"
    if (-not $success) {
        Write-Host "⚠️ 기본 패키지 동기화 실패" -ForegroundColor Yellow
    }
    
    return $success
}

# Python 버전 확인
function Test-PythonVersion {
    Write-Host "🔍 Python 버전 확인 중..." -ForegroundColor Yellow
    
    try {
        $pythonPath = Get-Command "python.exe" -ErrorAction SilentlyContinue
        if (-not $pythonPath) {
            Write-Host "❌ Python이 설치되지 않았습니다." -ForegroundColor Red
            Write-Host "Python 3.11 또는 3.12를 설치 후 다시 실행해주세요." -ForegroundColor Yellow
            exit 1
        }
        
        $pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        Write-Host "✅ Python 버전: $pythonVersion" -ForegroundColor Green
        
        if ($pythonVersion -eq "3.12") {
            Write-Host "✅ Python 3.12 감지됨 - 호환 패키지 버전 사용" -ForegroundColor Green
            return $true
        } elseif ($pythonVersion -eq "3.11") {
            Write-Host "✅ Python 3.11 감지됨 - 표준 패키지 버전 사용" -ForegroundColor Green
            return $false
        } else {
            Write-Host "⚠️ Python $pythonVersion 감지됨" -ForegroundColor Yellow
            Write-Host "   권장 버전: Python 3.11 또는 3.12" -ForegroundColor Yellow
            $continue = Read-Host "계속 진행하시겠습니까? (y/n)"
            if ($continue -ne 'y' -and $continue -ne 'Y') {
                Write-Host "설치를 중단합니다." -ForegroundColor Red
                exit 1
            }
            return $true
        }
    }
    catch {
        Write-Host "❌ Python 버전 확인 중 오류: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

# Python 3.12 호환 CPU 버전 설치
function Install-Python312CPUVersion {
    Write-Host "`n🖥️ Python 3.12 호환 CPU 패키지 설치 중..." -ForegroundColor Green
    
    # 기존 TensorFlow 제거 (충돌 방지)
    Write-Host "🧹 기존 TensorFlow 제거 중..." -ForegroundColor Yellow
    Invoke-UVCommand "uv remove tensorflow tensorflow-cpu" | Out-Null
    
    $packages = @(
        "tensorflow>=2.16.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0", 
        "torchaudio>=2.1.0",
        "xgboost>=2.0.3",
        "lightgbm>=4.1.0",
        "catboost>=1.2.2",
        "shap>=0.45.0",
        "statsmodels>=0.14.0",
        "plotly>=5.15.0",
        "wordcloud>=1.9.0",
        "openpyxl>=3.1.0",
        "nltk>=3.8.0",
        "Pillow>=10.0.0"
    )
    
    $success = $true
    foreach ($package in $packages) {
        $result = Invoke-UVCommand "uv add '$package'"
        if (-not $result) {
            $success = $false
        }
        Start-Sleep -Milliseconds 500
    }
    
    return $success
}

# CPU 버전 설치 (Python 3.11용)
function Install-CPUVersion {
    Write-Host "`n🖥️ 표준 CPU 전용 ML/DL 패키지 설치 중..." -ForegroundColor Green
    
    $packages = @(
        "tensorflow-cpu>=2.15.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "torchaudio>=2.0.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "catboost>=1.2.0",
        "shap>=0.45.0",
        "statsmodels>=0.14.0",
        "plotly>=5.15.0",
        "wordcloud>=1.9.0",
        "openpyxl>=3.1.0",
        "nltk>=3.8.0",
        "Pillow>=10.0.0"
    )
    
    $success = $true
    foreach ($package in $packages) {
        $result = Invoke-UVCommand "uv add '$package'"
        if (-not $result) {
            $success = $false
        }
        Start-Sleep -Milliseconds 500
    }
    
    return $success
}

# GPU 버전 설치 (Python 3.11용)
function Install-GPUVersion {
    Write-Host "`n🚀 표준 GPU 가속 ML/DL 패키지 설치 중..." -ForegroundColor Green
    
    $packages = @(
        "tensorflow[and-cuda]>=2.15.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0", 
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "catboost>=1.2.0",
        "shap>=0.45.0",
        "statsmodels>=0.14.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "plotly>=5.15.0",
        "wordcloud>=1.9.0",
        "openpyxl>=3.1.0",
        "nltk>=3.8.0",
        "Pillow>=10.0.0"
    )
    
    $success = $true
    foreach ($package in $packages) {
        $result = Invoke-UVCommand "uv add '$package'"
        if (-not $result) {
            $success = $false
        }
        Start-Sleep -Milliseconds 500
    }
    
    return $success
}

# Python 3.12 호환 GPU 버전 설치
function Install-Python312GPUVersion {
    Write-Host "`n🚀 Python 3.12 호환 GPU 패키지 설치 중..." -ForegroundColor Green
    
    # 기존 패키지 제거
    Write-Host "🧹 기존 패키지 제거 중..." -ForegroundColor Yellow
    Invoke-UVCommand "uv remove tensorflow tensorflow-cpu torch torchvision torchaudio" | Out-Null
    
    $packages = @(
        "tensorflow>=2.16.0",        # CPU/GPU 자동 감지
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "torchaudio>=2.1.0", 
        "xgboost>=2.0.3",
        "lightgbm>=4.1.0",
        "catboost>=1.2.2",
        "shap>=0.45.0",
        "statsmodels>=0.14.0",
        "transformers>=4.36.0",      # Python 3.12 호환
        "datasets>=2.16.0",          # Python 3.12 호환
        "accelerate>=0.25.0",        # Python 3.12 호환
        "plotly>=5.15.0",
        "wordcloud>=1.9.0",
        "openpyxl>=3.1.0",
        "nltk>=3.8.0",
        "Pillow>=10.0.0"
    )
    
    $success = $true
    foreach ($package in $packages) {
        $result = Invoke-UVCommand "uv add '$package'"
        if (-not $result) {
            $success = $false
        }
        Start-Sleep -Milliseconds 500
    }
    
    return $success
}

# 전체 버전 설치
function Install-FullVersion {
    Write-Host "`n🎯 전체 패키지 설치 중..." -ForegroundColor Green
    
    # GPU 패키지 먼저 설치
    $gpuSuccess = Install-GPUVersion
    
    # 추가 연구/MLOps 패키지
    $additionalPackages = @(
        "optuna>=3.2.0",
        "mlflow>=2.5.0",
        "wandb>=0.15.0",
        "bokeh>=3.1.0",
        "altair>=5.0.0",
        "holoviews>=1.17.0"
    )
    
    $additionalSuccess = $true
    foreach ($package in $additionalPackages) {
        $result = Invoke-UVCommand "uv add '$package'"
        if (-not $result) {
            $additionalSuccess = $false
        }
        Start-Sleep -Milliseconds 500
    }
    
    return ($gpuSuccess -and $additionalSuccess)
}

# 설정 파일 생성
function New-ConfigFile {
    param([string]$InstallMode)
    
    $configContent = @"
"""
AI Data Scientist 환경 설정 (Windows)
설치 모드: $InstallMode
설치 시간: $(Get-Date)
OS: $($env:OS)
"""

import platform
import sys

# 환경 설정
INSTALL_MODE = "$InstallMode"
CUDA_AVAILABLE = False
TENSORRT_AVAILABLE = False
OS_TYPE = "Windows"

def get_environment_info():
    """환경 정보를 반환합니다."""
    info = {
        'install_mode': INSTALL_MODE,
        'os': platform.system(),
        'python_version': sys.version,
        'packages': [],
        'gpu_support': False
    }
    
    # TensorFlow 확인
    try:
        import tensorflow as tf
        info['packages'].append(f'tensorflow=={tf.__version__}')
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            info['gpu_support'] = True
            global CUDA_AVAILABLE
            CUDA_AVAILABLE = True
            info['gpu_devices'] = [device.name for device in gpu_devices]
    except ImportError:
        pass
    
    # PyTorch 확인
    try:
        import torch
        info['packages'].append(f'torch=={torch.__version__}')
        if torch.cuda.is_available():
            info['gpu_support'] = True
            info['torch_cuda_devices'] = torch.cuda.device_count()
    except ImportError:
        pass
    
    # 기타 패키지들 확인
    packages_to_check = ['xgboost', 'lightgbm', 'catboost', 'shap', 'sklearn', 'pandas', 'numpy']
    for pkg in packages_to_check:
        try:
            module = __import__(pkg)
            if hasattr(module, '__version__'):
                info['packages'].append(f'{pkg}=={module.__version__}')
            else:
                info['packages'].append(f'{pkg}=installed')
        except ImportError:
            pass
    
    return info

def print_system_info():
    """시스템 정보를 출력합니다."""
    info = get_environment_info()
    print("🖥️ AI Data Scientist 환경 정보 (Windows):")
    print(f"   OS: {info['os']}")
    print(f"   Python: {info['python_version']}")
    print(f"   설치 모드: {info['install_mode']}")
    print(f"   GPU 지원: {'✅' if info['gpu_support'] else '❌'}")
    print(f"   설치된 패키지: {len(info['packages'])}개")
    for pkg in info['packages']:
        print(f"     - {pkg}")
    
    if 'gpu_devices' in info:
        print(f"   TensorFlow GPU 디바이스:")
        for device in info['gpu_devices']:
            print(f"     - {device}")
    
    if 'torch_cuda_devices' in info:
        print(f"   PyTorch CUDA 디바이스: {info['torch_cuda_devices']}개")

def check_gpu_availability():
    """GPU 가용성 상세 체크"""
    print("🔍 GPU 가용성 상세 체크:")
    
    # NVIDIA-SMI 확인
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi.exe'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA-SMI 사용 가능")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version' in line:
                    print(f"   CUDA 버전: {line.strip()}")
                    break
        else:
            print("❌ NVIDIA-SMI 사용 불가")
    except Exception as e:
        print(f"❌ NVIDIA-SMI 체크 실패: {e}")
    
    # TensorFlow GPU 체크
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow 버전: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"   GPU 디바이스: {len(gpus)}개")
        for i, gpu in enumerate(gpus):
            print(f"     {i}: {gpu.name}")
            # 메모리 정보 시도
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"       메모리 증가 모드 설정 완료")
            except Exception as e:
                print(f"       메모리 설정 실패: {e}")
    except Exception as e:
        print(f"❌ TensorFlow GPU 체크 실패: {e}")
    
    # PyTorch CUDA 체크
    try:
        import torch
        print(f"✅ PyTorch 버전: {torch.__version__}")
        print(f"   CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA 버전: {torch.version.cuda}")
            print(f"   디바이스 개수: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"     디바이스 {i}: {props.name}")
                print(f"       메모리: {props.total_memory / 1024**3:.1f} GB")
    except Exception as e:
        print(f"❌ PyTorch CUDA 체크 실패: {e}")

if __name__ == "__main__":
    print_system_info()
    print()
    check_gpu_availability()
"@

    Set-Content -Path "ai_config.py" -Value $configContent -Encoding UTF8
    Write-Host "📄 ai_config.py 설정 파일이 생성되었습니다." -ForegroundColor Green
}

# GPU 유틸리티 모듈 생성
function New-GPUUtilsModule {
    $gpuUtilsContent = @"
"""
Windows용 GPU 감지 및 최적화 모듈
"""

def check_gpu_availability():
    """Windows 환경에서 GPU 가용성을 확인합니다."""
    import subprocess
    import sys
    
    gpu_status = {
        'nvidia_smi': False,
        'tensorflow_gpu': False,
        'torch_gpu': False,
        'gpu_devices': [],
        'recommendations': []
    }
    
    # NVIDIA-SMI 확인
    try:
        result = subprocess.run(['nvidia-smi.exe', '--list-gpus'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_status['nvidia_smi'] = True
            gpu_status['gpu_devices'].extend(result.stdout.strip().split('\n'))
            print("✅ NVIDIA-SMI 사용 가능")
        else:
            print("❌ NVIDIA-SMI 사용 불가")
    except Exception as e:
        print(f"❌ NVIDIA-SMI 체크 실패: {e}")
    
    # TensorFlow GPU 확인
    try:
        import tensorflow as tf
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            gpu_status['tensorflow_gpu'] = True
            print(f"✅ TensorFlow GPU 사용 가능: {len(gpu_devices)}개 디바이스")
        else:
            print("⚠️ TensorFlow GPU 사용 불가")
            gpu_status['recommendations'].append("GPU 버전 TensorFlow 설치 고려")
    except ImportError:
        print("❌ TensorFlow 설치되지 않음")
        gpu_status['recommendations'].append("TensorFlow 설치 필요")
    
    # PyTorch GPU 확인
    try:
        import torch
        if torch.cuda.is_available():
            gpu_status['torch_gpu'] = True
            gpu_count = torch.cuda.device_count()
            print(f"✅ PyTorch CUDA 사용 가능: {gpu_count}개 디바이스")
        else:
            print("⚠️ PyTorch CUDA 사용 불가")
            gpu_status['recommendations'].append("CUDA 지원 PyTorch 설치 고려")
    except ImportError:
        print("❌ PyTorch 설치되지 않음")
    
    return gpu_status

def optimize_for_environment():
    """Windows 환경에 맞게 설정을 최적화합니다."""
    gpu_status = check_gpu_availability()
    
    # TensorFlow 메모리 증가 설정
    if gpu_status['tensorflow_gpu']:
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("🔧 TensorFlow GPU 메모리 증가 모드 설정 완료")
        except Exception as e:
            print(f"⚠️ TensorFlow GPU 설정 중 오류: {e}")
    
    return gpu_status

if __name__ == "__main__":
    print("🔍 Windows GPU 환경 확인 중...")
    status = check_gpu_availability()
    print("\n🛠️ 환경 최적화 적용 중...")
    optimize_for_environment()
    
    if status['recommendations']:
        print("\n💡 권장사항:")
        for rec in status['recommendations']:
            print(f"  - {rec}")
"@
    
    Set-Content -Path "gpu_utils.py" -Value $gpuUtilsContent -Encoding UTF8
    Write-Host "📄 gpu_utils.py 모듈 생성 완료" -ForegroundColor Green
}

# 설치 검증
function Test-Installation {
    Write-Host "`n🔍 설치 검증 중..." -ForegroundColor Yellow
    
    $testScript = @"
import sys
print(f'Python 버전: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')

packages = ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn']
missing = []

print('\n📋 패키지 설치 확인:')
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {pkg}=={version}')
    except ImportError:
        print(f'❌ {pkg}')
        missing.append(pkg)

# TensorFlow 확인 (Python 3.12 호환성 체크)
try:
    import tensorflow as tf
    print(f'✅ tensorflow=={tf.__version__}')
    if hasattr(tf.config, 'list_physical_devices'):
        gpu_count = len(tf.config.list_physical_devices('GPU'))
        print(f'   GPU 디바이스: {gpu_count}개')
        if sys.version_info >= (3, 12) and tf.__version__ >= '2.16.0':
            print('   ✅ Python 3.12 호환성 확인됨')
        elif sys.version_info >= (3, 12):
            print('   ⚠️ Python 3.12이지만 TensorFlow 버전이 낮을 수 있음')
    else:
        print('   ⚠️ GPU 디바이스 확인 불가')
except ImportError:
    print('❌ tensorflow')
    missing.append('tensorflow')

# PyTorch 확인
try:
    import torch
    print(f'✅ torch=={torch.__version__}')
    if torch.cuda.is_available():
        print(f'   CUDA 사용 가능: {torch.cuda.device_count()}개 디바이스')
        print(f'   CUDA 버전: {torch.version.cuda}')
    else:
        print('   CUDA 사용 불가 (CPU 모드)')
except ImportError:
    print('❌ torch')
    missing.append('torch')

# XGBoost, LightGBM 등 확인
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
    if sys.version_info >= (3, 12):
        print('✅ Python 3.12 환경에서 정상 동작 확인!')
"@
    
    try {
        $testResult = python -c $testScript
        Write-Host $testResult
        return $LASTEXITCODE -eq 0
    }
    catch {
        Write-Host "❌ 설치 검증 중 오류 발생: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# 사용법 안내
function Show-Usage {
    Write-Host "`n🎯 사용법 안내:" -ForegroundColor Green
    Write-Host "1. Streamlit 앱 시작:" -ForegroundColor White
    Write-Host "   streamlit run Home.py" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "2. 환경 정보 확인:" -ForegroundColor White
    Write-Host "   python ai_config.py" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "3. GPU 테스트:" -ForegroundColor White
    Write-Host "   python gpu_utils.py" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "4. 패키지 업데이트:" -ForegroundColor White
    Write-Host "   uv sync" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "5. PowerShell 실행 정책 문제 시:" -ForegroundColor White
    Write-Host "   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Cyan
    Write-Host ""
}

# 모드 선택 함수
function Select-InstallationMode {
    param([string]$PreferredMode, [object]$GPUInfo)
    
    if ($PreferredMode -eq "auto") {
        Write-Host "`n📋 설치 옵션을 선택하세요:" -ForegroundColor Yellow
        Write-Host "1) 🖥️  CPU 전용 (안정적, 모든 환경)" -ForegroundColor White
        Write-Host "2) 🚀 GPU 가속 (NVIDIA GPU 필요)" -ForegroundColor White
        Write-Host "3) 🔧 자동 감지 (추천)" -ForegroundColor White
        Write-Host "4) 🧪 연구용 전체 패키지" -ForegroundColor White
        Write-Host ""
        
        $choice = Read-Host "선택하세요 (1-4, 기본값: 3)"
        if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "3" }
        
        switch ($choice) {
            "1" { return "cpu" }
            "2" { return "gpu" }
            "3" { 
                if ($GPUInfo.HasNvidia) {
                    Write-Host "GPU가 감지되었습니다. GPU 버전을 설치하시겠습니까? (y/n, 기본값: y): " -ForegroundColor Yellow -NoNewline
                    $gpuChoice = Read-Host
                    if ([string]::IsNullOrWhiteSpace($gpuChoice) -or $gpuChoice -eq "y" -or $gpuChoice -eq "Y") {
                        return "gpu"
                    } else {
                        return "cpu"
                    }
                } else {
                    Write-Host "GPU가 감지되지 않았습니다. CPU 모드로 설치합니다." -ForegroundColor Yellow
                    return "cpu"
                }
            }
            "4" { return "full" }
            default { 
                Write-Host "⚠️ 잘못된 선택입니다. 자동 감지로 설치합니다..." -ForegroundColor Yellow
                return Select-InstallationMode -PreferredMode "auto" -GPUInfo $GPUInfo
            }
        }
    } else {
        return $PreferredMode
    }
}

# 메인 함수
function Main {
    # 도움말 표시
    if ($Help) {
        Show-Help
        return
    }
    
    # 헤더 표시
    Write-Header
    
    try {
        # 1. Python 버전 확인
        $usePy312Packages = Test-PythonVersion
        
        # 2. uv 확인 및 설치
        Test-UV
        
        # 3. GPU 감지
        $gpuInfo = Test-GPU
        
        # 4. 설치 모드 선택
        $installMode = Select-InstallationMode -PreferredMode $Mode -GPUInfo $gpuInfo
        Write-Host "`n🎯 선택된 설치 모드: $installMode" -ForegroundColor Green
        Write-Host "🎯 Python 3.12 호환 패키지: $usePy312Packages" -ForegroundColor Green
        
        # 5. 기본 패키지 설치
        Write-Host "`n📦 3단계: 기본 패키지 설치" -ForegroundColor Green
        $baseSuccess = Install-BasePackages
        
        # 6. 모드별 패키지 설치
        Write-Host "`n🧠 4단계: ML/DL 패키지 설치" -ForegroundColor Green
        $installSuccess = $false
        
        if ($usePy312Packages) {
            # Python 3.12 호환 패키지 사용
            switch ($installMode) {
                "cpu" {
                    Write-Host "🖥️ Python 3.12 호환 CPU 전용 설치를 시작합니다..." -ForegroundColor Yellow
                    $installSuccess = Install-Python312CPUVersion
                }
                "gpu" {
                    Write-Host "🚀 Python 3.12 호환 GPU 가속 설치를 시작합니다..." -ForegroundColor Yellow
                    $installSuccess = Install-Python312GPUVersion
                }
                "full" {
                    Write-Host "🧪 Python 3.12 호환 연구용 전체 패키지 설치를 시작합니다..." -ForegroundColor Yellow
                    $installSuccess = Install-Python312GPUVersion
                    # 추가 패키지
                    $additionalPackages = @(
                        "optuna>=3.5.0",
                        "mlflow>=2.9.0",
                        "wandb>=0.16.0",
                        "bokeh>=3.3.0"
                    )
                    foreach ($pkg in $additionalPackages) {
                        Invoke-UVCommand "uv add '$pkg'" | Out-Null
                    }
                }
                default {
                    Write-Host "❌ 알 수 없는 설치 모드: $installMode" -ForegroundColor Red
                    return
                }
            }
        } else {
            # 기존 패키지 버전 사용 (Python 3.11)
            switch ($installMode) {
                "cpu" {
                    Write-Host "🖥️ 표준 CPU 전용 설치를 시작합니다..." -ForegroundColor Yellow
                    $installSuccess = Install-CPUVersion
                }
                "gpu" {
                    Write-Host "🚀 표준 GPU 가속 설치를 시작합니다..." -ForegroundColor Yellow
                    $installSuccess = Install-GPUVersion
                }
                "full" {
                    Write-Host "🧪 표준 연구용 전체 패키지 설치를 시작합니다..." -ForegroundColor Yellow
                    $installSuccess = Install-FullVersion
                }
                default {
                    Write-Host "❌ 알 수 없는 설치 모드: $installMode" -ForegroundColor Red
                    return
                }
            }
        }
        
        # 7. 설정 파일 생성
        Write-Host "`n🛠️ 5단계: 설정 파일 생성" -ForegroundColor Green
        New-ConfigFile -InstallMode $installMode
        New-GPUUtilsModule
        
        # 8. 설치 검증
        Write-Host "`n🔍 6단계: 설치 검증" -ForegroundColor Green
        $verificationSuccess = Test-Installation
        
        # 9. 결과 출력
        Write-Host "`n" + "=" * 60 -ForegroundColor Yellow
        if ($baseSuccess -and $installSuccess -and $verificationSuccess) {
            if ($usePy312Packages) {
                Write-Host "🎉 Python 3.12 호환 AI Data Scientist 환경 설정이 완료되었습니다!" -ForegroundColor Green
            } else {
                Write-Host "🎉 AI Data Scientist 환경 설정이 완료되었습니다!" -ForegroundColor Green
            }
            Show-Usage
        } else {
            Write-Host "❌ 설치 중 일부 오류가 발생했습니다." -ForegroundColor Red
            Write-Host "로그를 확인하고 수동으로 누락된 패키지를 설치해주세요." -ForegroundColor Yellow
            Write-Host "" -ForegroundColor Yellow
            Write-Host "🔧 수동 설치 명령어:" -ForegroundColor Yellow
            if ($usePy312Packages) {
                Write-Host "   uv add 'tensorflow>=2.16.0' 'torch>=2.1.0' 'xgboost>=2.0.3'" -ForegroundColor Cyan
            } else {
                Write-Host "   uv add 'tensorflow-cpu>=2.15.0' 'torch>=2.0.0' 'xgboost>=2.0.0'" -ForegroundColor Cyan
            }
        }
        
    }
    catch {
        Write-Host "❌ 스크립트 실행 중 오류 발생: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "스택 트레이스: $($_.ScriptStackTrace)" -ForegroundColor Yellow
    }
}

# 스크립트 실행
Main