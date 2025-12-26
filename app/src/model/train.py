from ultralytics import YOLO


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

        print(f"学習完了\n学習済みモデル: runs/detect/deer_training/weights/best.pt")

        return results

    # def detect_image(image_path, model_path="runs/detect/deer_training/weights/best.pt"):
    #     """
    #     学習済みモデルで画像から鹿を検出する
    #     """
    #     # 学習済みモデルが存在しない場合は事前学習済みモデルを使用
    #     if not os.path.exists(model_path):
    #         print(f"学習済みモデルが見つかりません: {model_path}")
    #         print("事前学習済みモデル（yolov8n.pt）を使用します（鹿専用ではありません）")
    #         model_path = "yolov8n.pt"

    #     model = YOLO(model_path)

    #     # 推論実行
    #     results = model(image_path, conf=0.3)  # 信頼度30%以上

    #     # 結果を画像に描画
    #     for result in results:
    #         # 検出結果をプロット
    #         img_with_detections = result.plot()

    #         # 結果を保存
    #         output_path = f"detection_result_{os.path.basename(image_path)}"
    #         cv2.imwrite(output_path, img_with_detections)
    #         print(f"検出結果を保存: {output_path}")

    #         # 検出された物体の情報を表示
    #         if len(result.boxes) > 0:
    #             for box in result.boxes:
    #                 class_id = int(box.cls)
    #                 confidence = float(box.conf)
    #                 class_name = model.names[class_id]
    #                 print(f"検出: {class_name} (信頼度: {confidence:.2f})")
    #         else:
    #             print("鹿は検出されませんでした")
