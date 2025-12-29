# OpenCV Deer Detector

YOLOv8 モデルを使用して、画像やカメラ映像から鹿を検出するプロジェクトです。

## 概要

このプロジェクトは、Python と Ultralytics YOLO を使用して、鹿（Female deer, fawn, male-deer, unknown-deer）を検出します。
学習、テスト、およびリアルタイムカメラ検出の機能を提供します。

## 機能

- **モデル学習**: カスタムデータセットを使用して YOLOv8 モデルを学習。
- **モデルテスト**: テスト画像セットを使用してモデルの精度を評価。
- **リアルタイム検出**: Web カメラなどの映像入力からリアルタイムで鹿を検出し、結果を表示・保存。

## 必要要件

- Python 3.x
- Docker (オプション)
- CUDA 対応 GPU (推奨) 

## インストール

### ローカル環境

必要なライブラリをインストールします。

```bash
pip install -r requirements.txt
```

### Docker 環境

Docker Compose を使用してビルド・実行できます。
**リソースが限られている場合はDockerを使わないで直接動かす事を推奨します。**

```bash
docker-compose up --build
```

## 使い方

### 実行

```bash
python main.py
```

デフォルトでは、学習済みモデルを読み込み、カメラからのリアルタイム検出を開始します。
検出結果の動画は `output/` ディレクトリに保存されます。

### 設定

`app/src/config.py` で設定を変更できます。

- `RUN_MODE`: 実行モード
- `MODEL_PATH`: モデルのパス
- `DEVICE`: 使用デバイス (cpu, cuda, mps)

### 学習・テスト

`main.py` 内のコメントアウトを解除することで、学習やテストを実行できます。

```python
# TrainModel.train(cfd) # モデルの学習
# TrainModel.test_model(model_path=model_path, cfd=cfd) # モデルのテスト
```

## ディレクトリ構成

- `app/src/`: ソースコード
- `learning-media/`: データセット (画像とラベル)
- `runs/`: 学習結果 (モデルの重みなど)
- `output/`: 検出結果の動画保存先
- `tmp/`: テスト結果の画像保存先

## AI の使用について

このプロジェクトのコードおよびドキュメントの一部は、AI アシスタントを使用して生成・作成されています。
