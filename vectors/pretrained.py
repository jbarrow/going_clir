from collections import OrderedDict, defaultdict
from torchtext.data import Dataset, Field
from torchtext.vocab import _default_unk_index, Vocab, Vectors

import os

class MultiCCA(Vectors):

    url_base = 'https://s3.amazonaws.com/material-data/multiCCA_512.'

    def __init__(self, languages=['en', 'fr'], **kwargs):
        url = self.url_base + '.'.join(languages)
        name = os.path.basename(url)
        super(MultiCCA, self).__init__(name, url=url, **kwargs)

class VectorVocab(Vocab):
    def __init__(self, vectors, min_freq=1, specials=['<pad>'],
                 unk_init=None, vectors_cache=None):
        """
        Override the Vocab `__init__` function to ensure that `itos` and `stoi`
        are kept in line with the pretrained vectors passed in.
        """

        self.itos = list(specials) + vectors.itos
        self.stoi = defaultdict(_default_unk_index)
        
        self.stoi.update({w: i+len(list(specials)) for w, i in vectors.stoi.items()})

        self.vectors = vectors
        self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)

class VectorVocabField(Field):
    vocab_cls = VectorVocab
    
    def build_vocab(self, *args, **kwargs):
        """
        Overriding the Field `build_vocab` function to not include the
        counter -- the entirety of our vocab comes from the vectors.
        """
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        self.vocab = self.vocab_cls(specials=specials, **kwargs)
