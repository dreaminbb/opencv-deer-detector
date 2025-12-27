# YOLOv8を使用した鹿検出システム
# [] 動画（カメラ）から検出を行う
# [] 検出した動画をリアルタイムで表示、保存する
from app.src.config import Config
from app.src.model.train import TrainModel


def main():
    cfd = Config.initialize()
    model_path = "runs/detect/deer_training5/weights/best.pt"
    # TrainModel.train(cfd)
    TrainModel.test_model(model_path=model_path, cfd=cfd)


if __name__ == "__main__":
    main()
