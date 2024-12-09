import pandas as pd
# import numpy as np
# import h5py
import sys
import glob

argvs = sys.argv
file_tag =  sys.argv[1] 
file_tag = file_tag.split('raw_')[0]
h5_files_pattern = r'outnpy/{}*.h5py'.format(file_tag)
# print(h5_files_pattern)
h5_files = glob.glob(h5_files_pattern)
all_data = pd.DataFrame()
for files in h5_files:
    df = pd.read_hdf(files, key='winfo')
    all_data = pd.concat([all_data, df], ignore_index=True)
output_file = r'outnpy/{}.h5py'.format(file_tag[:-1])
all_data.to_hdf(output_file, key='winfo', mode='w')
print(all_data.shape)


