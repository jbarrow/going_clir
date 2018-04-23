import os
import json
import torch
import logging

class Checkpoint(object):
    def __init__(self, model, params={}, path='./', basename='model', save_every=10):
        self.path, self.basename, self.save_every = path, basename, save_every
        self.model = model
        self.current = 0
        self.best = 0.
        
        if not os.path.isdir(self.path):
            logging.warn(f'Model directory does not exist, creating: {self.path}')
            os.makedirs(self.path)
           
        with open(os.path.join(path, f'{basename}_params.json'), 'w') as fp:
            json.dump(params, fp)
        
    def update(self, metric=None):
        to_save = False
        filename = None
        
        if metric is not None and metric > self.best:
            filename = f'{self.basename}_best.pkl'
            to_save = True
            self.best = metric
           
        self.current += 1
        if self.current % self.save_every == 0:
            filename = f'{self.basename}_epoch_{self.current}.pkl'
            to_save = True
           
        if to_save and filename:
            filename = os.path.join(self.path, filename)
            logging.info(f'Saving model at epoch {self.current} to {filename}')
            torch.save(self.model.state_dict(), filename)

def load_model(cls, path, basename='model', best=True):
    model_filename = f'{basename}_best.pkl'
    with open(os.path.join(path, f'{basename}_params.json')) as fp:
        params = json.load(fp)

    model = cls(**params)
    model.load_state_dict(torch.load(os.path.join(path, model_filename)))
    
    return model