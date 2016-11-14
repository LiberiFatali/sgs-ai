import os, sys



WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
SYNSET_FILE = os.path.join(WORKING_DIR, 'sgsnet_synsets.txt')
METADATA_FILE = os.path.join(WORKING_DIR, 'sgsnet_metadata.txt')

# Create label->synset mapping
synsets = []
with open(SYNSET_FILE) as f:
  synsets = f.read().splitlines()
# Create synset->metadata mapping
texts = {}
with open(METADATA_FILE) as f:
  for line in f.read().splitlines():
    parts = line.split('\t')
    assert len(parts) == 2
    texts[parts[0]] = parts[1]
    
for s in synsets:
  print texts[s]