import os

root = '/fs/clip-scratch/jdbarrow'

def pathify(path, filename=None):
    if filename:
        path = os.path.join(path, filename)
    return os.path.join(root, path)