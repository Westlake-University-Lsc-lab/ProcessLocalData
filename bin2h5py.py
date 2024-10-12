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
        rawfilename = line.rstrip('\n') # [17 :] 
        rawdata = daw_readout.DAWDemoWaveParser(rawfilename)     
        for wave in tqdm(rawdata) :
            ch = wave.Channel
            ttt = wave.Timestamp
            base = wave.Baseline
            pulse = wave.Waveform
            st, minp, ed = process_data.pusle_index(pulse)
            ht = base - wave.Waveform[minp]
            area_fix_range = process_data.pulse_area_fix_range(pulse, 90, 140, base)
            area_fix_range_dy = process_data.pulse_area_fix_range(pulse, 90, 200, base)
            if ch == 0:  ### LV2414 Anode
                ht = ht * 2.76
                area_fix_range_pe = area_fix_range / 77.791 *2.76
            if ch == 1:
                area_fix_range_pe = area_fix_range / 96.99
                area_fix_range_pe = area_fix_range / 96.99
            if ch == 2:   #### LV2414 Dynode
                area_fix_range_pe = area_fix_range_dy / 77.791
            winfo.append({
                'Ch':ch,
                'TTT':ttt,   ## Trigger time tag
                'Baseline': base, 
                'Hight': ht, 
                'Area_fixrange':area_fix_range_pe,
                'Wave': pulse
            })
        #file_tag = line.rstrip('\n')[17 :].rstrip('.bin')[24:][: -12]  
        file_tag = line.rstrip('\n').rstrip('.bin')[24:][: -12]  
        path_save = "outnpy/{}.h5py".format(file_tag)
        df = pd.DataFrame(winfo)
        process_data.write_to_hdf5(df, path_save)
#print(path_save)
#data_array = df.values
#np.save(path_save, data_array)
