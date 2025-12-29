from app.src.config import Config
from app.src.module.model import DetectDeerWithModel


def main():
    cfd = Config.initialize()

    model_path = "runs/detect/deer_training5/weights/best.pt"
    # TrainModel.train(cfd) # モデルの学習
    # TrainModel.test_model(model_path=model_path, cfd=cfd) # モデルのテスト、大量の画像を認識させて、その後tmp/<timestamp>/に保存する
    DetectDeer = DetectDeerWithModel(model_path=model_path, cfd=cfd)
    DetectDeer.deploy_display_window_source_from_frame_via_opencv(
        is_show_video_realtime=True, camera_index=0
    )  # 動画（カメラ）からの検出を行う


if __name__ == "__main__":
    main()
