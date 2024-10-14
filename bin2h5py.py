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

gain_lv2414 = 77.791   # convertered to ADC  
gain_lv2415 = 96.99    # convertered to ADC  
attenuation_factor_9DB = 2.76   # attenuation factor 
attenuation_factor_6DB = 1.87   # attenuation factor 
attenuation_factor_12DB = 3.85  # attenuation factor 
attenuation_factor_18DB = 7.62   # attenuation factor 

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
            area_fix_range = process_data.pulse_area_fix_range(pulse, 90, 3000, base)
            #area_fix_range_dy = process_data.pulse_area_fix_range(pulse, 90, 300, base)
            if ch == 0:  ### LV2414 Anode
                ht = ht * attenuation_factor_9DB
                area_fix_range_pe = area_fix_range / gain_lv2414 *attenuation_factor_9DB
            if ch == 1:
                area_fix_range_pe = area_fix_range / gain_lv2415
                area_fix_range_pe = area_fix_range / gain_lv2415
            if ch == 2:   #### LV2414 Dynode
                area_fix_range_pe = area_fix_range / gain_lv2414
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
