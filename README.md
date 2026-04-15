# 🧪 MLOps Demo — YOLOv8 + WandB Sweep

YOLOv8 객체 탐지 모델을 기반으로 **WandB(Weights & Biases)** 를 활용한 실험 추적 및 **Bayesian Hyperparameter Sweep** 자동화를 실습하는 MLOps 데모 프로젝트입니다.

---

## 📁 파일 구조

```
mlops_demo/
├── .env                  # WandB API 키 등 환경 변수 (Git에 포함하지 않도록 주의)
├── requirements.txt      # 프로젝트 의존 패키지 목록
├── sweep.yaml            # WandB Sweep 설정 파일 (탐색 전략 및 하이퍼파라미터 범위 정의)
├── train_baseline.py     # 베이스라인 단일 학습 스크립트
├── train_sweep.py        # WandB Sweep 연동 학습 스크립트
├── yolov8n.pt            # 사전 학습된 YOLOv8n 모델 가중치
├── datasets/             # 학습 데이터셋 (자동 다운로드 또는 직접 배치)
├── runs/                 # YOLO 학습 결과물 저장 디렉토리
└── wandb/                # WandB 로컬 로그 디렉토리
```

---

## ⚙️ 환경 설정

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. WandB 설정

프로젝트 루트에 `.env` 파일을 생성하고 WandB API 키를 설정합니다.

```
# .env
WANDB_API_KEY=your_wandb_api_key_here
```

> **API 키 발급**: [https://wandb.ai/authorize](https://wandb.ai/authorize) 에서 확인

`.env` 파일은 `python-dotenv`가 자동으로 로드하며, 코드 내에서 `load_dotenv()`를 호출해 환경 변수를 읽어옵니다.

> ⚠️ `.env` 파일은 절대 Git에 커밋하지 마세요. `.gitignore`에 반드시 추가하세요.

---

## 📊 WandB 설정 설명

두 학습 스크립트 모두 WandB를 실험 추적 도구로 사용합니다.

### 베이스라인 (`train_baseline.py`)

```python
wandb.init(project="mlops_demo_yolo", name="baseline_run", job_type="training")
```

| 파라미터 | 설명 |
|---|---|
| `project` | WandB 프로젝트 이름 |
| `name` | 이 실험 Run의 이름 |
| `job_type` | 실험 유형 분류 태그 (`training`) |

- 매 에포크 종료 시 `val_batch0_pred.jpg`(검증 추론 이미지)와 `results.png`(Loss/Metric 곡선)를 자동 로깅합니다.

### Sweep 학습 (`train_sweep.py`)

```python
with wandb.init(project="mlops_demo_yolo", job_type="sweep", config=default_config):
    config = wandb.config
```

- `wandb.init()`을 **context manager** 방식으로 사용합니다.
- `config=default_config`로 기본 하이퍼파라미터를 설정하며, Sweep 에이전트가 이를 자동으로 덮어씁니다.
- `wandb.config`를 통해 Sweep이 주입한 하이퍼파라미터를 학습 함수에 전달합니다.
- 매 에포크마다 `on_fit_epoch_end` 콜백으로 Loss/Metric과 검증 이미지를 함께 로깅합니다.

---

## 🔍 Sweep 설정 설명 (`sweep.yaml`)

```yaml
program: train_sweep.py        # Sweep 에이전트가 실행할 스크립트
method: bayes                  # 탐색 전략: Bayesian Optimization
metric:
  goal: maximize               # 목표: 지표 최대화
  name: metrics/mAP50-95(B)   # 최적화 대상 지표

parameters:
  batch:
    values: [8, 16, 32]        # 배치 크기 후보
  epochs:
    values: [10, 15, 20]       # 에포크 수 후보
  imgsz:
    values: [320, 480, 640]    # 입력 이미지 크기 후보
  lr0:
    distribution: uniform      # 균등 분포로 연속 탐색
    min: 0.0001
    max: 0.01
  weight_decay:
    distribution: uniform
    min: 0.0001
    max: 0.001
```

| 항목 | 설명 |
|---|---|
| `method: bayes` | 이전 실험 결과를 바탕으로 다음 탐색 지점을 지능적으로 선택하는 Bayesian Optimization 사용 |
| `goal: maximize` | 목표 지표를 최대화하는 방향으로 탐색 |
| `name: metrics/mAP50-95(B)` | mAP@50:95 (박스 기준) 를 최적화 목표 지표로 설정 |
| `values: [...]` | 이산형 후보값 목록에서 선택 |
| `distribution: uniform` | 연속형 파라미터를 균등 분포 범위 내에서 샘플링 |

---

## 💻 코드 설명

### `train_baseline.py` — 베이스라인 단일 학습

| 구성 요소 | 설명 |
|---|---|
| `load_dotenv()` | `.env` 파일에서 `WANDB_API_KEY` 로드 |
| `wandb.init()` | WandB 실험 세션 시작 |
| `YOLO("yolov8n.pt")` | 사전 학습된 YOLOv8n 모델 로드 |
| `model.add_callback(...)` | 에포크 종료 시 커스텀 로깅 콜백 등록 |
| `model.train(...)` | COCO128 데이터셋으로 학습 실행 |
| `log_custom_plots()` | 검증 예측 이미지와 결과 곡선을 WandB에 업로드 |
| `wandb.finish()` | WandB 세션 정상 종료 |

### `train_sweep.py` — Sweep 연동 학습

| 구성 요소 | 설명 |
|---|---|
| `settings.update({'clearml': False})` | ClearML 자동 연동 비활성화 (WandB만 사용) |
| `with wandb.init(..., config=default_config)` | context manager 방식으로 Sweep Run 초기화 |
| `wandb.config` | Sweep 에이전트가 주입한 하이퍼파라미터 접근 |
| `model.add_callback(...)` | 에포크 종료 시 Loss + 검증 이미지 로깅 콜백 등록 |
| `log_epoch_metrics()` | `trainer.metrics`(Loss/mAP)와 `val_batch0_pred.jpg` 이미지를 `wandb.log()`로 업로드 |

---

## 🚀 실행 방법

### 1. 베이스라인 학습

단일 고정 하이퍼파라미터로 기준 성능을 측정합니다.

```bash
python train_baseline.py
```

학습이 완료되면 WandB 대시보드에서 `mlops_demo_yolo` 프로젝트의 `baseline_run` 결과를 확인할 수 있습니다.

---

### 2. Sweep 실행 (하이퍼파라미터 자동 탐색)

**Step 1: Sweep 생성**

`sweep.yaml`을 WandB에 등록하여 Sweep ID를 발급받습니다.

```bash
wandb sweep sweep.yaml
```

실행 결과 예시:
```
wandb: Creating sweep from: sweep.yaml
wandb: Created sweep with ID: abc12345
wandb: View sweep at: https://wandb.ai/<your-entity>/mlops_demo_yolo/sweeps/abc12345
```

**Step 2: Sweep 에이전트 실행**

발급받은 Sweep ID로 에이전트를 실행합니다. 에이전트는 Sweep 설정에 따라 `train_sweep.py`를 반복 실행합니다.

```bash
wandb agent <your-entity>/<project-name>/<sweep-id>
```

> 실행 횟수 제한을 두려면 `--count` 옵션을 사용하세요.
> ```bash
> wandb agent --count 10 <your-entity>/<project-name>/<sweep-id>
> ```

**Step 3: 결과 확인**

WandB 대시보드의 Sweep 탭에서 각 Run의 성능을 비교하고 최적 하이퍼파라미터를 확인합니다.

---

## 📦 의존 패키지

```
ultralytics      # YOLOv8 학습 프레임워크
wandb            # 실험 추적 및 Sweep 플랫폼
opencv-python    # 이미지 처리
python-dotenv    # .env 파일 기반 환경 변수 관리
```
