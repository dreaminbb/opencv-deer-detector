import os
import cv2


class SystemHandler:
    """
    システム関連の操作を行う
    カメラの取得を行うが、その他映像関連の操作もここで行う
    映像の保管や、取得、削除、移動など
    """

    @staticmethod
    def fetch_camera(camera_index=0):
        """
        カメラを取得する
        基本カメラは１つしかないので、デフォルトは０にしている
        :param camera_index: カメラのインデックス
        :return: cv2.VideoCaptureオブジェクト
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"カメラが開けません。インデックス: {camera_index}")
        return cap

    # テスト、カメラの映像をリアルタイムで表示
    @staticmethod
    def display_camera_feed_realtime(
        camera_index=0, ret=None, frame_from_source=None, window_name="Camera Feed"
    ):
        """
        カメラの映像をリアルタイムで表示する
        :param camera_index: カメラのインデックス
        :param frame_from_source: カメラから取得したフレーム
        """

        ret, frame = ret, frame_from_source
        if not ret:
            print("カメラから映像を取得できませんでした。")
            return

        cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(window_name, frame)


    @staticmethod
    def save_video_from_frames(frames: list, output_path: str, fps: int = 30):

        if len(frames) == 0:
            print("保存するフレームがありません。")
            return
        else:
            height, width, layers = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            if os.path.exists(output_path):
                print(f"既に同名のファイルが存在します。上書きします: {output_path}")
            else:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for frame in frames:
                video_writer.write(frame)

            video_writer.release()
            print(f"動画を保存しました: {output_path}")
