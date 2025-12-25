import argparse


# 設定のクラス
# これはmain.py実行時に初めにインスタンス化、実行される
class Config:

    CONFIG = {
        "RUN_MODE": "training",
        "MODEL_PATH": "models/deer_detector.onnx",
        "DATA_CONFIG_PATH": "data.yaml",
        "DEVICE": "gpu",
        # 画像処理関連
        "IMAGE_SIZE": 640,
    }

    def __init__(self):
        pass

    @staticmethod
    def initialize():
        config = Config()
        config.fetch_run_arguments()
        return config

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
