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
            # st, minp, ed = process_data.pusle_index(pulse)
            # ht = base - wave.Waveform[minp]
            # area_s1 = process_data.pulse_area_fix_range(pulse, 50, 300, base)
            # area_s2 = process_data.pulse_area_fix_range(pulse, 1250, 1750, base)    ### 5us
            # area_s2 = process_data.pulse_area_fix_range(pulse, 2500, 3000, base)   ### 10us
            # area_s2 = process_data.pulse_area_fix_range(pulse, 7500, 8000, base)    ### 30us
            # area_s2 = process_data.pulse_area_fix_range(pulse, 22500,23000, base)    ### 90us
            # area_s2 = process_data.pulse_area_fix_range(pulse, 50000,50800, base)   ### 200us
            area = process_data.pulse_area_fix_range(pulse, 80, 380, base)
            if ch == 0:
                area = area / gain_lv2414 *attenuation_factor_12DB
                # area_S1 = area_s1 / gain_lv2414 *attenuation_factor_12DB
                # area_S2 = area_s2 / gain_lv2414 *attenuation_factor_12DB
            if ch == 1:
                area = area / gain_lv2415 * attenuation_factor_6DB
                # area_S1 = area_s1 / gain_lv2415 *attenuation_factor_6DB
                # area_S2 = area_s2 / gain_lv2415 *attenuation_factor_6DB
            if ch == 2:
                area = area / gain_lv2414
                # area_S1 = area_s1 / gain_lv2414 
                # area_S2 = area_s2 / gain_lv2414 
            winfo.append({         
                'Ch':ch,
                'TTT':ttt,   ## Trigger time tag
                'Baseline': base, 
                # 'Area_S1':area_S1,
                # 'Area_S2':area_S2,
                'Area_S2':area,
                # 'S1_width':152, #ns
                'S2_width':920, #ns  5us, 90us, 200us
                # 'S2_width':1070, #ns  10us
                # 'S2_width':1080, #ns  30us
                # 'Delta_t': 200, #us
                'Wave': pulse
            })    
        #file_tag = line.rstrip('\n')[17 :].rstrip('.bin')[24:][: -12]  
        file_tag = line.rstrip('\n').rstrip('.bin')[24:][:]  
        path_save = "outnpy/{}.h5py".format(file_tag)
        df = pd.DataFrame(winfo)
        process_data.write_to_hdf5(df, path_save)
#print(path_save)
#data_array = df.values
#np.save(path_save, data_array)
