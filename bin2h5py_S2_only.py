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
processed_list = []
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
        delt_t = rawfilename.split('680mv_')[1].split('us_50hz')[0]
        rawdata = daw_readout.DAWDemoWaveParser(rawfilename)     
        for wave in tqdm(rawdata) :
            ch = wave.Channel
            ttt = wave.Timestamp
            base = wave.Baseline
            base_l20 = np.mean(wave.Waveform[ -20:])
            if ch == 0 :
                if delt_t == '1': 
                    base = base_l20                  
                if delt_t == '2' : 
                    base = base_l20
                # print(r'this ch={}, baseline:{}, base_l20:{}'.format(ch, base, base_l20))                  
            pulse = wave.Waveform
            # st, minp, ed = process_data.pusle_index(pulse)
            # ht = base - wave.Waveform[minp]
            # area_s1 = process_data.pulse_area_fix_range(pulse, 50, 300, base)
            # area_s2 = process_data.pulse_area_fix_range(pulse, 1250, 1750, base)    ### 5us
            area = process_data.pulse_area_fix_range(pulse, 100, 370, base)
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
                'TTT':ttt,   
                'Baseline': base, 
                'base_l20': base_l20,               
                'Area_S2':area,
                'S1_width':150, #ns
                'S2_width':1000, #ns  5us, 90us, 200us
                'Delta_t': int(delt_t), #us
                'Wave': pulse
            })    
        #file_tag = line.rstrip('\n')[17 :].rstrip('.bin')[24:][: -12]  
        file_tag = line.rstrip('\n').rstrip('.bin')[24:][:]  
        path_save = "outnpy/{}.h5py".format(file_tag)
        processed_list.append(path_save)
        df = pd.DataFrame(winfo)
        process_data.write_to_hdf5(df, path_save)
#print(path_save)
#data_array = df.values
#np.save(path_save, data_array)

fited_file=file_list+'_fit'
with open(fited_file, 'w') as f:
    for s in processed_list:
        f.write(s + '\n')
    print('Fited file saved to {}'.format(fited_file))