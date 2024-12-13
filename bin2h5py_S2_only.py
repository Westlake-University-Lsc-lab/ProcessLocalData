import daw_readout
import process_data
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import sys
import time
import os
import constant

argvs = sys.argv
file_list =  sys.argv[1] 
processed_list = []
path_save =""


gain_lv2414 = constant.gain_lv2414 
gain_lv2415 = constant.gain_lv2415 
attenuation_factor_9DB = constant.attenuation_factor_9DB 
attenuation_factor_6DB = constant.attenuation_factor_6DB 
attenuation_factor_12DB = constant.attenuation_factor_12DB 
attenuation_factor_18DB = constant.attenuation_factor_18DB 
attenuation_factor_20DB = constant.attenuation_factor_20DB 

with open(file_list, 'r') as list:
    for line in list:  
        winfo =[]
        rawfilename = line.rstrip('\n') # [17 :] 
        delta_t = rawfilename.split('680mv_')[1].split('us_50hz')[0]
        # voltage = rawfilename.split('20241212_')[1].split('v_S2_OFF')[0].replace('p','.')        
        # voltage = rawfilename.split('20241212_')[1].split('v_calibration')[0].replace('p','.')        
        rawdata = daw_readout.DAWDemoWaveParser(rawfilename)     
        for wave in tqdm(rawdata) :
            ch = wave.Channel
            ttt = wave.Timestamp
            base = wave.Baseline            
            if ch == 0 and delta_t == '1' :
                base = 15171
            if ch == 2 and delta_t == '1' :
                base =  8112
            pulse = wave.Waveform
            # area = process_data.pulse_area_fix_range(pulse, 50, 370, base) ### for charge calibration
            area = process_data.pulse_area_fix_range(pulse, 100, 370, base)   ## for S2 calibration
            # area = process_data.pulse_area_fix_range(pulse, 350, 620, base)   ## for dt=1us S2 calibration
            # area = process_data.pulse_area_fix_range(pulse, 600, 870, base)   ## for df=2us S2 calibration
            if ch == 0:
                # area = area / gain_lv2414 *attenuation_factor_20DB
                # area = area / gain_lv2414 *attenuation_factor_9DB
                area = area / gain_lv2414 *attenuation_factor_12DB
                # area = area / gain_lv2414              
            if ch == 1:
                # area = area / gain_lv2415 * attenuation_factor_12DB
                # area = area / gain_lv2415 * attenuation_factor_9DB
                area = area / gain_lv2415                
            if ch == 2:
                area = area / gain_lv2414
            winfo.append({         
                'Ch':ch,
                'TTT':ttt,   
                'Baseline': base, 
                'Area_S2':area,
                'S1_width':150, #ns
                'S2_width':1000, #ns 
                'Delta_t': int(delta_t), #us
                # 'Voltage': float(voltage),
                'Wave': pulse
            })    
        #file_tag = line.rstrip('\n')[17 :].rstrip('.bin')[24:][: -12]  
        file_tag = line.rstrip('\n').rstrip('.bin')[24:][:]  
        path_save = "outnpy/{}.h5py".format(file_tag)
        processed_list.append(path_save)
        df = pd.DataFrame(winfo)
        process_data.write_to_hdf5(df, path_save)

fited_file=file_list+'_fit'
with open(fited_file, 'w') as f:
    for s in processed_list:
        f.write(s + '\n')
    print('Fited file saved to {}'.format(fited_file))
