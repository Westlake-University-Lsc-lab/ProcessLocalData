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
winfo =[]
i=0
with open(file_list, 'r') as list:
    for line in list:    
        rawfilename = line.rstrip('\n')[17 :] 
        rawdata = daw_readout.DAWDemoWaveParser(rawfilename)     
        for wave in tqdm(rawdata) :
            ch = wave.Channel
            ttt = wave.Timestamp
            base = wave.Baseline
            pulse = wave.Waveform
            st, minp, ed = process_data.pusle_index(pulse)
            ht = base - wave.Waveform[minp]
            rst = minp  - st
            flt = ed - minp
            maxp = base - np.max(pulse)
            asy = ht / (ht + maxp)
            area = process_data.pusle_area(pulse, st, ed, base)
            area_fixlen = process_data.pulse_area_fix_len(pulse, st, 10, base)
            winfo.append({
                'EvtID': i,
                'Ch':ch,
                'TTT':ttt,   ## Trigger time tag
                'Baseline': base, 
                'Hight': ht, 
                'Width': ed -st,  ## pusle width
                'Minp': minp,     ## min point
                'Rst' : rst,      ## rise time of pulse    
                'Flt' : flt,      ## full time of pulse  
                'Asym' : asy,     ## asymmetry of pulse
                'Area': area,     ## area of pulse cal by dynamic range
                'AreaFixlen': area_fixlen,    ## area of pulse cal by fix length of 10
                'Wave': pulse,
            })
            i +=1
            
        file_tag = line.rstrip('\n')[17 :].rstrip('.bin')[24:][: -13]  
        path_save = "outnpy/{}.h5".format(file_tag)

df = pd.DataFrame(winfo)
process_data.write_to_hdf5(df, path_save)