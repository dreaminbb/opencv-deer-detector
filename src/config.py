import argparse
import yaml


# 設定のクラス
# これはmain.py実行時に初めにインスタンス化、実行される
class Config:

    CONFIG = {
        "RUN_MODE": "training",
        "MODEL_PATH": "models/deer_detector.onnx",
        "DATA_CONFIG_PATH": "data.yaml",
        "DEVICE": "gpu",
        # 画像処理関連 yamlファイルから読み込む
        "DATA_YAML_FILENAME": "docker_data.yaml",
        "IMAGE_SIZE": 640,
        "media_root": "",
        "train_image_dir": "",
        "val_image_dir": "",
        "test_image_dir": "",
        "nc": 0,
        "names": [],
    }

    def __init__(self):
        self.fetch_yaml_config()
        pass

    @staticmethod
    def initialize():
        config = Config()
        config.fetch_run_arguments()
        return config

    def fetch_yaml_config(self) -> dict:

        print("YAMLファイルを読み込み中。。。")
        # ファイル読み込み
        file = open("./docker_data.yaml", "r")
        config = yaml.safe_load(file)

        self.CONFIG["media_root"] = config["path"]
        self.CONFIG["train_image_dir"] = config["train"]
        self.CONFIG["val_image_dir"] = config["val"]
        self.CONFIG["test_image_dir"] = config["test"]
        self.CONFIG["nc"] = config["nc"]
        self.CONFIG["names"] = config["names"]
        file.close()
        print("YAMLファイルの読み込み完了")
        print(config)
        return

    def fetch_config(self) -> dict:
        return {
            "RUN_MODE": self.RUN_MODE,
            "MODEL_PATH": self.MODEL_PATH,
            "DATA_CONFIG_PATH": self.DATA_CONFIG_PATH,
            "IMAGE_SIZE": self.IMAGE_SIZE,
            "DEVICE": self.DEVICE,
        }

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

        opt = {
            "env": parser.parse_args().env,
            "device": parser.parse_args().device,
        }

        if opt["env"] not in ["training", "detection"]:
            print("そんな設定ねえよ")
            opt["env"] = "training"

        if opt["device"] not in ["cpu", "gpu"]:
            print("そんな設定ねえよ")
            opt["device"] = "gpu"

        self.CONFIG["RUN_MODE"] = (
            "detection" if opt["env"] == "detection" else "training"
        )
        self.CONFIG["DEVICE"] = opt["device"]

        debug_msg = (
            f"RUN_MODE: {self.CONFIG['RUN_MODE']}, DEVICE: {self.CONFIG['DEVICE']}"
        )
        print(debug_msg)

        return

    def fetch_config(self) -> dict:
        return self.CONFIG
