import json
import pickle
from pathlib import Path
import pandas as pd
'''This script creat scene_camera.json of test_nobop'''

# save_file_index = 'local_data/bop_datasets/lmo/index_test.feather'
# a = pd.read_feather(save_file_index).reset_index(drop=True)
# print(a)

bop_list = 'local_data/bop_datasets/lmo/image_set/lmo_bop_test.txt'
all_json=json.loads(Path('local_data/bop_datasets/lmo/test_all/000002/scene_camera.json').read_text())
with open(bop_list) as f:
    bop_list = f.readlines()

for id in bop_list:
    id = id.strip('\n')
    all_json.pop(id)

all_json = json.dumps(all_json, indent=2)
with open('asshold.json', 'w') as f:
    f.write(all_json)
