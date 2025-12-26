# TODO
# []　モデルを保存、読み込み
# []　学習率、エポック数、バッチサイズなどのハイパーパラメータを引数から指定できるようにする
# []　指定した画像から検出を行う
# []　動画から検出を行う
# []　検出した動画をリアルタイムで表示、保存する
from src.config import Config


def main():
    Config.initialize()
    config_dict = Config().fetch_config()
    print(config_dict)


if __name__ == "__main__":
    print("Start main.py")
    main()
