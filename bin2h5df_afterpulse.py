import daw_readout
import process_data
import runinfo as runinfo
from tqdm import tqdm
import sys
import time
import argparse
import numpy as np
import pandas as pd

import constant as c
atten_9DB = c.attenuation_factor_9DB 
atten_6DB = c.attenuation_factor_6DB 
atten_12DB = c.attenuation_factor_12DB 
atten_18DB = c.attenuation_factor_18DB 
atten_20DB = c.attenuation_factor_20DB 



def read_file_names(file_list_path):
    flist = []
    with open(file_list_path, 'r') as file:
        for line in file:
            rawfilename = line.rstrip('\n')  
            flist.append(rawfilename)  
    return flist

# process self trigger data
def process_batch(file, runtype):
    file_tag = runinfo.find_file_tag(file)        
    winfo =[]        
    rawdata = daw_readout.DAWDemoWaveParser(file)      
    for wave in tqdm(rawdata):
        ch = wave.Channel
        ttt = wave.Timestamp
        base = wave.Baseline
        data = wave.Waveform  
        # wlen = len(data)
        # std = np.std(data[:10])
        # rms = np.sqrt(np.mean(np.square(data[:20])))
        st,ed,md =process_data.pulse_index(data, base, 0.01, 7)      
        # area= process_data.pulse_area(data, st, ed, base)   
        area= process_data.pulse_area(data, 95, 105, base) 
        # if ch == 0 :
            # area = area * atten_6DB
        # elif ch == 1:    
            # area= area * atten_6DB      
        hight = base - data[md]
        width = ed - st 
        # rfhight = base - np.min(data[md+3:md+20])
        asys = (base - np.min(data))/(np.max(data) - np.min(data))
        # rfovhight = base - np.max(data[md:md+20])
        if runtype == 'DarkRate':
            winfo.append({
                'Ch':ch,
                'TTT':ttt,
                'Baseline': base, 
                # 'RMS':rms,
                # 'STD':std,
                'Area':area,
                'Hight':hight,
                'st':st,
                'ed':ed,
                'md':md,
                'width':width,
                'Asys':asys,             
                # 'WLen':wlen,
                'Wave': data,  
                })
    file_tag = file_tag
    print(file_tag)
    path_save = "/mnt/data/outnpy/{}.h5py".format(file_tag)
    df = pd.DataFrame(winfo)
    process_data.write_to_hdf5(df, path_save)  
    print(path_save)
    return   path_save
        

def main():
    try:
        parser = argparse.ArgumentParser(description='Process raw data to hdf5 format')
        parser.add_argument('--runtype', type=str, help='DarkRate')
        parser.add_argument('--file_list', type=str, help='file list')
        args = parser.parse_args()
        if len(vars(args)) != 2:
            raise Exception("Invalid number of arguments.")
        if args.runtype not in ["DarkRate","LED","others"]:
            raise Exception("Invalid runtype.")

        start_time = time.time()

        runtype = args.runtype
        file_list = args.file_list
        print("Arguments parsed successfully.")
        trig_mode = runinfo.check_trigger_mode(runtype)
        flist = read_file_names(file_list)
        processed_list = [] 
        for rawfilename in flist:                       
            if trig_mode == 'Self':
                processed_list.append(process_batch(rawfilename, runtype))
            else:
                print("Attention: Unknown trigger mode.")
                sys.exit(1)
        output_file = file_list + '_processed'
        with open(output_file, 'w') as file:
            for filename in processed_list:
                file.write(filename+'\n')
        print("Processed files saved to: ", output_file)
        
        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        print(f"\nTotal time: {minutes}m {seconds}s")
        
    except Exception as e:
        print("An error occurred while parsing arguments:", str(e))
        print("Usagee: python bin2h5df.py --runtype DarkRate --file_list file_list.txt")
        sys.exit(1)


if __name__ == '__main__':
    main()
    
    