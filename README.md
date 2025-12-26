# opencv-deer-detector

# 実行

## Docker、Windows（NVIDIA 　 GPU 搭載）環境

Docker 環境では GPU を使用するために実行時に引数を指定

```
# ビルド
docker build -t opencv-deer-detector .

# 実行 (GPUを使用する場合)
docker run --gpus all -it opencv-deer-detector
```
