import torch

class Config(object):

    def __init__(self, data_path):
        self.job_name = 'distilbert'
        self.name = 'best'
        self.split = 'distilbert'
        self.batch_size = 16
        self.num_workers = 4
        self.lr = 1e-5
        self.dropout = 0.25
        self.label_smoothing = 0.1
        self.concat_last_n = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generate_output = 1
