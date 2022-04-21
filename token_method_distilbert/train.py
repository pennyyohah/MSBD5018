from data_preprocessing import ToxicDataModule
from model_built import BertModel
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from args import Config
import os

os.environ["TOKENIZERS_PARALLELISM"] = 'true'

data_path = './data'

config = Config(data_path)

# Load data and models

toxic_data_module = ToxicDataModule(data_path, config.batch_size, config.num_workers)
model = BertModel(args=config)

# Logger
tb_logger = pl_loggers.TensorBoardLogger(save_dir='logs', name=config.job_name)

# Callbacks
callbacks = []

# Save best model checkpoints callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_f1',
    dirpath=os.path.join('logs', config.job_name, 'version_' + str(tb_logger.version)),
    filename='{epoch:02d}-{val_f1:.2f}',
    save_top_k=3,
    mode='max',
    save_weights_only=True,
    save_last=False)

callbacks.append(checkpoint_callback)

# Train the model
trainer = Trainer.from_argparse_args(config, logger=tb_logger, callbacks=callbacks)
trainer.fit(model, datamodule=toxic_data_module)