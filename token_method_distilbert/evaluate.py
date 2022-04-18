import os
from zipfile import ZipFile

os.environ["TOKENIZERS_PARALLELISM"] = 'true'

from pytorch_lightning import Trainer

from data_preprocessing import ToxicDataModule
from model_built import BertModel
from util import f1_score
from args import Config

data_path = './data'
config = Config(data_path)
def compute_ensemble_predictions(predictions):
    n_models = len(predictions)
    n_sentences = len(predictions[0])
    ensemble_predictions = []
    for i in range(n_sentences):
        counter = {}
        p = set()
        for j in range(n_models):
            for e in predictions[j][i]:
                counter[e] = counter.get(e, 0) + 1
                if counter[e] / n_models >= 0.5:
                    p.add(e)
        ensemble_predictions.append(p)
    return ensemble_predictions



weights_path = os.path.join('weights', config.name)
checkpoint_names = []
for file in os.listdir(weights_path):
    if file.endswith('.ckpt'):
        checkpoint_names.append(os.path.join(weights_path, file))

toxic_data_module = ToxicDataModule(data_path, config.batch_size, config.num_workers)

if len(checkpoint_names) > 1:
    print(f"Evaluating an ensemble of {len(checkpoint_names)} models:\n")

original_spans = None
predicted_spans = []
for i, checkpoint in enumerate(checkpoint_names):
    print(f"Evaluating model {checkpoint} ...")

    model = BertModel.load_from_checkpoint(checkpoint_path=checkpoint)

    toxic_data_module.setup(stage=config.split)
    test_loader = toxic_data_module.test_dataloader()

    trainer = Trainer.from_argparse_args(config, logger=False)
    trainer.test(model, test_dataloaders=test_loader, verbose=False)

    data = model.predictions
    predicted_spans.append(data['predicted_spans'])
    if i == 0:
        original_spans = data['original_spans']

    f1 = f1_score(data['predicted_spans'], original_spans)
    print(f"Result for model {checkpoint} --> f1-score = {f1:.4f}\n")

if len(checkpoint_names) > 1:
    predicted_spans_ensemble = compute_ensemble_predictions(predicted_spans)
    f1 = f1_score(predicted_spans_ensemble, original_spans)
    print(f"\nResult for ensemble --> f1-score = {f1:.4f}")
else:
    predicted_spans_ensemble = predicted_spans[0]

if config.generate_output:
    zip_file = os.path.join(weights_path, 'output.zip')
    out_name = 'spans-pred.txt'
    out_file = os.path.join(weights_path, out_name)
    with open(out_file, 'w') as f:
        for i, offs in enumerate(predicted_spans_ensemble):
            f.write(f"{str(i)}\t{str(sorted(offs))}\n")
    with ZipFile(zip_file, 'w') as f:
        f.write(out_file, arcname=out_name)



