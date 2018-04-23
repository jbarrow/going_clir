import pandas as pd
from utils import pathify

data = pd.read_csv(pathify('data/SICK.txt'), sep='\t')

for dataset in set(data['SemEval_set']):
    data[data['SemEval_set'] == dataset].to_csv(pathify('data/sick', f'{dataset.lower()}.txt'), sep='\t')