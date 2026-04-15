import os
import random
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO, settings

# ClearML 자동 연동 해제 (WandB만 사용하기 위함)
settings.update({'clearml': False})
import wandb

# ─────────────────────────────────────────────────
# 전역 상태: 콜백에서 wandb.run 및 설정값을 참조하기 위함
# ─────────────────────────────────────────────────
_log_img_every_n_batch: int = 20  # 기본값; wandb.config에서 덮어씌워짐
_batch_log_count: int = 0          # 현재 학습 배치 카운터


def log_train_batch_images(trainer):
    """
    학습 배치가 끝날 때마다, N 배치 주기로 배치 내 이미지 일부를
    추론하여 WandB에 로깅합니다.
    - 원본 배치 이미지(gt labels)를 wandb.Image로 시각화
    """
    global _batch_log_count
    _batch_log_count += 1

    # N 배치마다 한 번씩만 로깅
    if _batch_log_count % _log_img_every_n_batch != 0:
        return

    save_dir = Path(trainer.save_dir)
    epoch = trainer.epoch

    # YOLO가 저장하는 train_batch*.jpg 파일들을 찾아 로깅
    batch_imgs = sorted(save_dir.glob("train_batch*.jpg"))
    if not batch_imgs:
        return

    # 최대 3개 배치 이미지 파일만 로깅 (용량 절약)
    imgs_to_log = batch_imgs[:3]
    wandb_images = [
        wandb.Image(str(p), caption=f"Train Batch | Epoch {epoch} | {p.name}")
        for p in imgs_to_log
    ]

    wandb.log({
        "Train/batch_images": wandb_images,
        "trainer/batch_count": _batch_log_count,
    }, commit=False)


def log_val_predictions(trainer):
    """
    에포크 학습이 끝날 때마다:
    1) Loss 및 metrics 로깅
    2) 검증 배치 추론 결과 이미지(val_batch*_pred.jpg) 로깅
    3) 검증 GT 이미지(val_batch*_labels.jpg) 비교 로깅
    4) results.png (전체 학습 곡선) 로깅
    """
    log_dict = {"epoch": trainer.epoch + 1}

    # ── 1. metrics / loss 수집 ──────────────────────
    if hasattr(trainer, "metrics") and trainer.metrics:
        log_dict.update(trainer.metrics)
    if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
        # loss_items는 Tensor 일 수 있으므로 float 변환
        try:
            loss_names = getattr(trainer, "loss_names", [])
            for name, val in zip(loss_names, trainer.loss_items):
                log_dict[f"train/{name}"] = float(val)
        except Exception:
            pass

    save_dir = Path(trainer.save_dir)

    # ── 2. 검증 예측 이미지 ─────────────────────────
    pred_imgs = sorted(save_dir.glob("val_batch*_pred.jpg"))
    if pred_imgs:
        log_dict["Val/predictions"] = [
            wandb.Image(str(p), caption=f"Pred | Epoch {trainer.epoch + 1} | {p.name}")
            for p in pred_imgs[:3]
        ]

    # ── 3. 검증 GT(라벨) 이미지 ──────────────────────
    label_imgs = sorted(save_dir.glob("val_batch*_labels.jpg"))
    if label_imgs:
        log_dict["Val/ground_truth"] = [
            wandb.Image(str(p), caption=f"GT | Epoch {trainer.epoch + 1} | {p.name}")
            for p in label_imgs[:3]
        ]

    # ── 4. 전체 결과 곡선 이미지 (results.png) ────────
    results_path = save_dir / "results.png"
    if results_path.exists():
        log_dict["Charts/results_plot"] = wandb.Image(
            str(results_path),
            caption=f"Results up to Epoch {trainer.epoch + 1}"
        )

    wandb.log(log_dict)


def log_model_summary(model, config):
    """
    학습 시작 전, 모델 아키텍처 요약 정보를 WandB에 Table로 기록합니다.
    다른 sweep run과 비교할 수 있도록 모델명 / 파라미터 수 / 입력 크기를 남깁니다.
    """
    try:
        n_params = sum(p.numel() for p in model.model.parameters())
        table = wandb.Table(
            columns=["model", "n_params_M", "imgsz", "batch", "lr0", "weight_decay", "epochs"],
            data=[[
                config.model,
                round(n_params / 1e6, 2),
                config.imgsz,
                config.batch,
                config.lr0,
                config.weight_decay,
                config.epochs,
            ]]
        )
        wandb.log({"Model/architecture_summary": table})
    except Exception as e:
        print(f"[warn] log_model_summary 실패: {e}")


def main():
    global _log_img_every_n_batch, _batch_log_count
    load_dotenv()

    # ── 기본 하이퍼파라미터 (Sweep이 없을 때 폴백) ──────
    default_config = {
        'model': 'yolov8n.pt',
        'lr0': 0.01,
        'weight_decay': 0.0005,
        'batch': 16,
        'epochs': 10,
        'imgsz': 640,
        'log_img_every_n_batch': 20,
    }

    with wandb.init(
        project="mlops_demo_yolo",
        job_type="sweep",
        config=default_config,
    ):
        config = wandb.config

        # 이미지 로깅 주기 설정
        _log_img_every_n_batch = config.log_img_every_n_batch
        _batch_log_count = 0

        # ── 모델 로드 (Sweep 파라미터로 모델 선택) ─────────
        model_name = config.model
        model = YOLO(model_name)
        print(f"[sweep] 모델: {model_name}")

        # wandb run name에 모델 이름 포함 (W&B UI에서 구분 편리)
        wandb.run.name = f"{model_name.replace('.pt','')}_{wandb.run.id}"

        # ── 모델 요약 로깅 ───────────────────────────────
        log_model_summary(model, config)

        # ── 콜백 등록 ────────────────────────────────────
        # 학습 배치 단위: 일정 주기로 배치 이미지 로깅
        model.add_callback("on_train_batch_end", log_train_batch_images)
        # 에포크 단위: val 추론 이미지 + metrics 로깅
        model.add_callback("on_fit_epoch_end", log_val_predictions)

        # ── 학습 실행 ────────────────────────────────────
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

        # ── 학습 완료 후: confusion matrix / PR curve 이미지 로깅 ──
        save_dir = Path(model.trainer.save_dir)
        final_artifacts = {
            "confusion_matrix.png": "Charts/confusion_matrix",
            "PR_curve.png":         "Charts/PR_curve",
            "F1_curve.png":         "Charts/F1_curve",
            "R_curve.png":          "Charts/R_curve",
            "P_curve.png":          "Charts/P_curve",
        }
        for fname, log_key in final_artifacts.items():
            fpath = save_dir / fname
            if fpath.exists():
                wandb.log({log_key: wandb.Image(str(fpath), caption=fname)})

        # ── 최종 모델 artifact 업로드 ────────────────────
        best_pt = save_dir / "weights" / "best.pt"
        if best_pt.exists():
            artifact = wandb.Artifact(
                name=f"yolo-best-{wandb.run.id}",
                type="model",
                description=f"Best weights from sweep run | model={model_name}",
                metadata=dict(config),
            )
            artifact.add_file(str(best_pt))
            wandb.log_artifact(artifact)
            print(f"[sweep] best.pt artifact 업로드 완료: {best_pt}")


if __name__ == "__main__":
    main()
