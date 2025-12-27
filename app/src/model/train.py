import time
import cv2
from ultralytics import YOLO
import os


class TrainModel:

    # 学習させる
    @staticmethod
    def train(cfd: object) -> object:
        """
        データセットを使用してPytorchを用いてYOLOモデルを学習する
        """

        message = f"\n=== 学習開始 ===\nデータセット設定ファイル: {cfd['MEDIA_ROOT']}\n学習パラメータ: epochs={cfd['EPOCHS']}, batch_size={cfd['BATCH_SIZE']}\n"
        print(message)

        # YOLOv8モデルをロード（初回は自動ダウンロード）
        model = YOLO(f"yolov8{cfd['YOLO_MODEL_SIZE']}.pt")  # nano版（軽量）

        #     # 学習実行
        results = model.train(
            data=cfd["DATA_CONFIG_PATH"],
            epochs=cfd["EPOCHS"],
            batch=cfd["BATCH_SIZE"],
            imgsz=cfd["IMAGE_SIZE"],
            project="runs/detect",
            name="deer_training",
            save_period=10,
            val=True,
            verbose=True,
            device=cfd["DEVICE"],
            workers=cfd["WORKERS"],
        )

        return results

    @staticmethod
    def test_model(model_path: str, cfd: object) -> None:
        """
        学習済みのモデルからテストを行う
        モデルのテストを行う、learning_madia/test/にテスト用の画像が入っている想定で実行する
        もしファイルが無かったらexit(1)で強制的に終了させる
        """

        # パスの確認
        if model_path == "":
            model_path = cfd["MODEL_DEFAULT_PATH"]
            print(
                f"モデルパスが指定されていません。デフォルトのモデルパスを使用します。\n{model_path}"
            )

        if not os.path.exists(model_path):
            print(f"学習済みモデルが見つかりません: {model_path}")
            exit(1)
        else:
            print(f"学習済みモデルを使用します: {model_path}")

        model = YOLO(model_path)

        # Docker環境のパス設定を使用
        test_images_dir = os.path.join(cfd["MEDIA_ROOT"], "test", "images")
        print(test_images_dir, cfd["MEDIA_ROOT"])

        # ローカル環境での実行を考慮して、パスが存在しない場合は相対パスを確認する
        if not os.path.exists(test_images_dir):
            # プロジェクトルートからの相対パス（learning-media/test/images）を確認
            local_path = os.path.join("learning-media", "test", "images")
            if os.path.exists(local_path):
                print(
                    f"Configのパスが見つからないため、ローカルパスを使用します: {local_path}"
                )
                test_images_dir = local_path

        if not os.path.exists(test_images_dir):
            print(f"テスト画像ディレクトリが見つかりません: {test_images_dir}")
            exit(1)

        # テスト画像を読みんで検出を行う
        # その後、正答率を計算して、表示する

        answer_count = 0
        problem_count = len(os.listdir(test_images_dir))
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(cfd["TEST_RESULT_SAVE_PATH"], timestamp)

        for image_file in os.listdir(test_images_dir):

            print(
                f"--- テスト画像: {image_file} ---\n進捗：{answer_count+1}/{problem_count}"
            )

            image_path = os.path.join(test_images_dir, image_file)
            # 画像以外のファイルは無視する
            if not os.path.isfile(image_path):
                print(f"ファイルではないためスキップ: {image_file}")
                continue
            if not image_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                print(f"画像ファイルではないためスキップ: {image_file}")
                continue

            results = model(image_path, conf=0.3)  # 信頼度30%以上

            answer_count += 1

            print(f"検出結果: {image_file}")
            # 結果を画像に描画
            for result in results:
                # 検出結果をプロット
                img_with_detections = result.plot()

                # 結果を保存、保存する場所は、設定にある、TEST_RESULT_SAVE_PATHを使う
                if not os.path.exists(save_path):
                    print(f"保存先が見つからなかったので作成{save_path}")
                    os.makedirs(save_path)
                output_path = os.path.join(
                    save_path, f"detection_result_{os.path.basename(image_path)}"
                )
                cv2.imwrite(output_path, img_with_detections)
                print(f"検出結果を保存: {output_path}")

                # 検出された物体の情報を表示
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        confidence = float(box.conf)
                        class_name = model.names[class_id]
                        print(f"検出: {class_name} (信頼度: {confidence:.2f})")
                else:
                    print("鹿は検出されませんでした")
