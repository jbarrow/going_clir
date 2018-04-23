from trectools import TrecDocs

import os

root = '/fs/clip-scratch/jdbarrow'


if __name__ == '__main__':
    d1 = TrecDocs.from_file(os.path.join(root, 'CLEF/More_French/sda94'), text_tags=['ld', 'tx'], title_tag='ti')
    d2 = TrecDocs.from_file(os.path.join(root, 'CLEF/French_collection/raw'))
    d = d1 + d2

    d.docs.to_csv(os.path.join(root, 'docs.csv'))
