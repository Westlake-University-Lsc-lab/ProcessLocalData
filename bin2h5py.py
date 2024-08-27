import daw_readout
import process_data
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import sys
import time
import os

argvs = sys.argv
file_list =  sys.argv[1] 
path_save =""

with open(file_list, 'r') as list:
    for line in list:  
        winfo =[]
        rawfilename = line.rstrip('\n')[17 :] 
        rawdata = daw_readout.DAWDemoWaveParser(rawfilename)     
        for wave in tqdm(rawdata) :
            ch = wave.Channel
            ttt = wave.Timestamp
            base = wave.Baseline
            pulse = wave.Waveform
            st, minp, ed = process_data.pusle_index(pulse)
            ht = base - wave.Waveform[minp]
            area = process_data.pusle_area(pulse, st, ed, base)
            area_fix_len = process_data.pulse_area_fix_len(pulse, minp, 15, base)
            area_fix_range = process_data.pulse_area_fix_range(pulse, 95, 130, base)
            if ch == 0:
                area_pe = area / 77.791
                area_fix_len_pe = area_fix_len / 77.791
                area_fix_range_pe = area_fix_range / 77.791
            if ch == 1:
                area_pe = area / 96.99
                area_fix_len_pe = area_fix_len / 96.99
                area_fix_range_pe = area_fix_range / 96.99
            if ch == 2:
                area_pe = area 
                area_fix_len_pe = area_fix_len 
                area_fix_range_pe = area_fix_range
            winfo.append({
                'Ch':ch,
                'TTT':ttt,   ## Trigger time tag
                'Baseline': base, 
                'Hight': ht, 
                'Area': area_pe,     ## area of pulse cal by dynamic range
                'Area_fixlen': area_fix_len_pe,  ## area of pulse cal by fix length 10
                'Area_fixrange':area_fix_range_pe
                #'Wave': pulse,
            })
        file_tag = line.rstrip('\n')[17 :].rstrip('.bin')[24:][: -12]  
        path_save = "outnpy/{}.h5py".format(file_tag)
        df = pd.DataFrame(winfo)
        process_data.write_to_hdf5(df, path_save)
#print(path_save)
#data_array = df.values
#np.save(path_save, data_array)
