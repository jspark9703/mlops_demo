import os
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO, settings

# ClearML 자동 연동 해제 (WandB만 사용하기 위함)
settings.update({'clearml': False})
import wandb

def log_epoch_metrics(trainer):
    """
    에포크 학습이 끝날 때마다 Loss 및 생성된 검증 추론 이미지(Prediction 이미지)를 
    WandB에 명시적으로 로깅하는 콜백 함수입니다.
    """
    log_dict = {"epoch": trainer.epoch}
    
    # YOLO의 에포크 결과 (loss 포함) 추가
    if hasattr(trainer, "metrics") and trainer.metrics:
        log_dict.update(trainer.metrics)

    save_dir = Path(trainer.save_dir)
    # YOLO는 검증 배치의 첫 번째 예측 이미지를 val_batch0_pred.jpg로 저장합니다.
    pred_img_path = save_dir / "val_batch0_pred.jpg"
    
    if pred_img_path.exists():
        log_dict["Validation_Predictions"] = wandb.Image(
            str(pred_img_path), 
            caption=f"Epoch {trainer.epoch}"
        )
        
    wandb.log(log_dict)

def main():
    load_dotenv()
    
    # 기본 하이퍼파라미터 설정 (Sweep에 의해 덮어씌워짐)
    default_config = {
        'lr0': 0.01,
        'weight_decay': 0.0005,
        'batch': 16,
        'epochs': 10,
        'imgsz': 640
    }
    
    # wandb context manager 사용 및 파라미터 튜닝 추적
    with wandb.init(project="mlops_demo_yolo", job_type="sweep", config=default_config):
        config = wandb.config
        
        model = YOLO("yolov8n.pt")
        
        # 에포크마다 Validation Prediction 및 Loss(metrics)를 확인할 수 있도록 콜백 등록
        model.add_callback("on_fit_epoch_end", log_epoch_metrics)
        
        # Sweep으로 전달된 파라미터를 학습 함수에 적용
        model.train(
            data="coco128.yaml",
            epochs=config.epochs,
            imgsz=config.imgsz,
            plots=True,
            save=True,
            lr0=config.lr0,
            weight_decay=config.weight_decay,
            batch=config.batch,
        )

if __name__ == "__main__":
    main()
