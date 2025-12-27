import argparse
import yaml
import torch


# 設定のクラス
# これはmain.py実行時に初めにインスタンス化、実行される
class Config:

    _instance = None
    INITIALIZE_CONFIG = {}

    CONFIG = {
        "RUN_MODE": "training",
        "MODEL_PATH": "models/deer_detector.onnx",
        "DATA_CONFIG_PATH": "data.yaml",
        "DEVICE": "gpu",  # 計算デバイス設定、GPU,CUDA,MPS
        # 画像処理関連 yamlファイルから読み込む
        "DATA_YAML_FILENAME": "docker_data.yaml",
        "MEDIA_ROOT": "learning_media",
        "TRAIN_IMAGE_DIR": "learning_media/train/",
        "VAL_IMAGE_DIR": "learning_media/val/",
        "TEST_IMAGE_DIR": "learning_media/test/",
        "EPOCHS": 100,
        "BATCH_SIZE": 16,
        "NC": 0,
        "IMAGE_SIZE": 640,
        "NAMES": [],
        # モデル設定
        # モデル	ファイルサイズ	パラメータ数	推論速度	精度(mAP)	用途
        # YOLOv8n	6.2MB	3.2M	最速	37.3	軽量デバイス、リアルタイム => n
        # YOLOv8s	21.5MB	11.2M	高速	44.9	バランス型 => s
        # YOLOv8m	49.7MB	25.9M	中速	50.2	高精度重視 => m
        # YOLOv8l	83.7MB	43.7M	やや遅	52.9	高精度アプリ => l
        # YOLOv8x	136MB	68.2M	最遅	53.9	最高精度 => xA
        # モデル使用時のデフォルトパス
        "MODEL_DEFAULT_PATH": "runs/detect/deer_training/weights/best.pt",
        "YOLO_MODEL_SIZE": "n",
        # 検出した画像の保存先、tmp/test_results/<timestamp>/
        "TEST_RESULT_SAVE_PATH": "tmp/test_results/",
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        設定初期化時に、実行環境に応じた設定に変更する
        Device -> GPU
        Docker + Mac -> MPS
        Docker + Windows/NVIDIA GPU -> CUDA
        """

        self.fetch_yaml_config()
        self.fetch_run_arguments()
        self.INITIALIZE_CONFIG = self.CONFIG

        device = "gpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "gpu"

        self.CONFIG["DEVICE"] = device
        print(f"使用デバイス: {self.CONFIG['DEVICE']}\n")
        pass

    @classmethod
    def get_config(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance.CONFIG

    @classmethod
    def get(cls, key, default=None):
        config = cls.get_config()
        return config.get(key, default)

    @staticmethod
    def initialize():
        config = Config()
        return config.INITIALIZE_CONFIG

    def fetch_yaml_config(self) -> dict:

        print("YAMLファイルを読み込み中。。。\n")
        # ファイル読み込み
        file = open("./docker_data.yaml", "r", encoding="utf-8")
        config = yaml.safe_load(file)

        self.CONFIG["MEDIA_ROOT"] = config["path"]
        self.CONFIG["TRAIN_IMAGE_DIR"] = config["train"]
        self.CONFIG["VAL_IMAGE_DIR"] = config["val"]
        self.CONFIG["TEST_IMAGE_DIR"] = config["test"]
        self.CONFIG["NC"] = config["nc"]
        self.CONFIG["NAMES"] = config["names"]
        file.close()
        print("YAMLファイルの読み込み完了\n")
        return

    # 実行時に取得するオプション；全てのオプションではなく、RUN_MODEとDEVICEのみを取得する
    # 他のオプションはファイルで設定しておくので、取得する必要性はない
    def fetch_run_arguments(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--env", type=str, default="local", help="実行環境")
        parser.add_argument(
            "--device",
            type=str,
            default="gpu",
            help="計算デバイス指定、CPUかGPU、GPUを絶対に推奨する",
        )
        parser.add_argument("--debug", action="store_true", help="デバッグモード有効化")
        parser.add_argument(
            "--epochs", type=int, default=100, help="学習エポック数指定"
        )
        parser.add_argument(
            "--batch_size", type=int, default=16, help="バッチサイズ指定"
        )
        parser.add_argument(
            "--model_size", type=str, default="n", help="YOLOモデルサイズ指定"
        )
        parser.add_argument(
            "--workers", type=int, default=4, help="データローダーのワーカー数"
        )

        opt = {
            "env": parser.parse_args().env,
            "device": parser.parse_args().device,
            "epochs": parser.parse_args().epochs,
            "batch_size": parser.parse_args().batch_size,
            "model_size": parser.parse_args().model_size,
            "workers": parser.parse_args().workers,
        }

        self.CONFIG["RUN_MODE"] = (
            "detection" if opt["env"] == "detection" else "training"
        )
        self.CONFIG["DEVICE"] = opt["device"]

        self.CONFIG["EPOCHS"] = opt["epochs"]
        self.CONFIG["BATCH_SIZE"] = opt["batch_size"]
        self.CONFIG["YOLO_MODEL_SIZE"] = opt["model_size"]
        self.CONFIG["WORKERS"] = opt["workers"]

        debug_msg = (
            f"RUN_MODE: {self.CONFIG['RUN_MODE']}, DEVICE: {self.CONFIG['DEVICE']}"
        )
        print(debug_msg)

        return
