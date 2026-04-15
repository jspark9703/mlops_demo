import os
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO
import wandb

def log_custom_plots(trainer):
    """
    에포크 학습이 끝날 때마다 생성된 
    검증 추론 이미지 및 Loss 곡선 그래프(results.png)를 
    WandB에 명시적으로 로깅하는 콜백 함수입니다.
    """
    save_dir = Path(trainer.save_dir)
    
    # 1. 검증 세트 추론 결과 이미지 로깅
    pred_img_path = save_dir / "val_batch0_pred.jpg"
    if pred_img_path.exists():
        wandb.log({
            "Validation_Predictions": wandb.Image(
                str(pred_img_path), 
                caption=f"Epoch {trainer.epoch}"
            )
        }, commit=False)
        
    # 2. Loss 및 평가지표 곡선 이미지 (results.png) 로깅
    results_path = save_dir / "results.png"
    if results_path.exists():
        wandb.log({
            "Loss & Metrics Plot": wandb.Image(
                str(results_path),
                caption=f"Results Plot up to Epoch {trainer.epoch}"
            )
        }, commit=False)

def main():
    # 환경 변수 로드 (API 키 등)
    load_dotenv()
    
    # WandB 초기화 (베이스라인 학습용)
    wandb.init(project="mlops_demo_yolo", name="baseline_run", job_type="training")
    
    # 사전 학습된 경량 모델 로드
    model = YOLO("yolov8n.pt")
    
    # 커스텀 콜백 등록: 매 에포크마다 Loss Plot과 추론 이미지 로깅
    model.add_callback("on_fit_epoch_end", log_custom_plots)
    
    # 학습 실행 (plots=True로 Loss와 기본 성능 지표, 학습/검증 결과 이미지 자동 로깅)
    results = model.train(
        data="coco128.yaml", # COCO128 등 기본 데이터셋 (필요시 데이터 yaml 파일 경로 지정)
        epochs=10,
        imgsz=640,
        batch=16,
        plots=True,
        save=True
    )
    
    # WandB 종료
    wandb.finish()

if __name__ == "__main__":
    main()
