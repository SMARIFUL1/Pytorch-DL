#this one running now and need to explore this more like segmentation

from anomalib import TaskType
from anomalib.models import EfficientAd
from anomalib.engine import Engine
from anomalib.deploy import ExportType
from anomalib.callbacks import ModelCheckpoint
from anomalib.data import Folder


dataset_root = r"D:\seminar\dataset\train"

def train():
    # Create the datamodule
    datamodule = Folder(
        name="train",
        root=dataset_root,
        normal_dir="normal",
        abnormal_dir="anomaly",
        # mask_dir="abnormal_mask",
        train_batch_size=1,
        num_workers=0,
        task=TaskType.CLASSIFICATION
    )
    datamodule.setup()

    model = EfficientAd()
    engine = Engine(max_epochs=30, task=TaskType.CLASSIFICATION,
                    callbacks=[ModelCheckpoint(dirpath='checkpoint/', every_n_epochs=1, save_last=True)])

    # logger.info("Checkpoint path: {}".format(engine.trainer.default_root_dir))

    # Train the model
    engine.fit(datamodule=datamodule, model=model)
    engine.export(export_type=ExportType.OPENVINO,
                  model=model,
                  export_root='anomalib_weight')

if __name__ == "__main__":
    train()