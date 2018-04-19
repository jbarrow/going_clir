from trectools import TrecDocs

d1 = TrecDocs.from_file('../material/CLEF/More_French/sda94', text_tags=['ld', 'tx'], title_tag='ti')
d2 = TrecDocs.from_file('../material/CLEF/French_collection/raw')
d = d1 + d2

d.docs.to_csv('docs.csv')