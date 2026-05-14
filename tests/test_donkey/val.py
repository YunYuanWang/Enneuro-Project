import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'code'))

import time
import os
import numpy as np

from eneuro.nn.loss import meanSquaredError
from eneuro.train import Evaluator

from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive
from eneuro.utils.serializer import Serializer


model = ResNet18AutoDrive()
script_dir = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(script_dir, "results", "model.json")

serializer = Serializer()
serializer.load(model, model_folder)

val_dataset = AutoDriveDataset(mode="val", transform=preprocess_image)

from eneuro.data.dataloader import DataLoader
val_loader = DataLoader(
    val_dataset,
    batch_size=400,
    shuffle=True,
    drop_last=True,
)

loss_fn = meanSquaredError

evaluator = Evaluator(model, loss_fn)

start = time.time()
test_loss, test_acc = evaluator.evaluate(
    val_loader,
    batch_size=400,
    verbose=True
)

print(f"MSE: {test_loss:.6f}")
print(f"平均单张样本用时 {(time.time() - start) / len(val_dataset):.3f} 秒")
