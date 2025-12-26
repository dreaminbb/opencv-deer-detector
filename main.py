# YOLOv8を使用した鹿検出システム
# [] デバイスで学習を最適化させる（Windows -> CUDA、 Mac -> MPS）
# [] 指定した画像から検出を行う
# [] 動画から検出を行う
# [] 検出した動画をリアルタイムで表示、保存する
import asyncio
from app.src.config import Config
from app.src.model.train import TrainModel


async def main():
    cfd = Config.initialize()
    await TrainModel.train(cfd)


if __name__ == "__main__":
    asyncio.run(main())
