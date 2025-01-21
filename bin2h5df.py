import daw_readout
import process_data
import runinfo as runinfo
import pandas as pd
from tqdm import tqdm
import sys
# import time
import argparse
import constant as c
gain_2414 = c.gain_lv2414 
gain_2415 = c.gain_lv2415 
atten_9DB = c.attenuation_factor_9DB 
atten_6DB = c.attenuation_factor_6DB 
atten_12DB = c.attenuation_factor_12DB 
atten_18DB = c.attenuation_factor_18DB 
atten_20DB = c.attenuation_factor_20DB 
atten_0DB = 1.0

class RunInfo:
    def __init__(self, run_type, run_info):
        self.run_type = run_type
        self.run_info = run_info
   
def read_file_names(file_list_path):
    flist = []
    with open(file_list_path, 'r') as file:
        for line in file:
            rawfilename = line.rstrip('\n')  
            flist.append(rawfilename)  
    return flist

def process(rawfilename, runtype):
    # runtype = runinfo.determine_runtype(rawfilename)    
    file_run_info = runinfo.parse_run_info(rawfilename)
    run_info = file_run_info[0]        
    # raw_file_info = RunInfo(runtype, file_run_info)
    winfo =[]        
    rawdata = daw_readout.DAWDemoWaveParser(rawfilename)      
    for wave in tqdm(rawdata):
        ch = wave.Channel
        ttt = wave.Timestamp
        base = wave.Baseline
        pulse = wave.Waveform
        area = process_data.pulse_area_fix_range(pulse, 50, 370, base)
        if ch == 0 :
            if runtype == "TimeConstant":
                area = area / gain_2414 *atten_20DB
            elif runtype == "Saturation":
                area = area / gain_2414 *atten_9DB
            elif runtype == "LongS2":
                area = area / gain_2414 *atten_0DB            
            else:
                print("Attention: Unknown runtype, use default gain and attenuation factor 9DB")
                area = area / gain_2414 *atten_9DB
        elif ch == 1 :
            if runtype == "TimeConstant":
                area = area / gain_2415 *atten_9DB
            elif runtype == "Saturation":
                area = area / gain_2415 
            elif runtype == "LongS2":
              area = area / gain_2415*atten_0DB 
            else:
                print("Attention: Unknown runtype, use default gain and attenuation factor 0DB")
                area = area / gain_2415         
        elif ch == 2:
            area = area / gain_2414 
        winfo.append({
            'Ch':ch,
            'TTT':ttt,
            'Baseline': base, 
            'Area':area,
            'S1_width':150, #ns
            'S2_width':1000, #ns 
            'RunType': runtype,            
            'Delta_t': run_info['delta_t'],
            'Voltage': run_info['voltage'],
            'RunTag': run_info['run_tag'],
            'Wave': pulse,            
        })
    file_tag = run_info['file_tag']
    print(file_tag)
    path_save = "outnpy/{}.h5py".format(file_tag)
    df = pd.DataFrame(winfo)
    process_data.write_to_hdf5(df, path_save)  
    print(path_save)
    return   path_save

def main():
    try:
        parser = argparse.ArgumentParser(description='Process raw data to hdf5 format')
        parser.add_argument('--runtype', type=str, help='TimeConstant or Calibration or Saturation')
        parser.add_argument('--file_list', type=str, help='file list')
        args = parser.parse_args()
        if len(vars(args)) != 2:
            raise Exception("Invalid number of arguments.")
        if args.runtype not in ["TimeConstant", "Calibration", "Saturation"]:
            raise Exception("Invalid runtype.")
        runtype = args.runtype
        file_list = args.file_list
        print("Arguments parsed successfully.")
        
        flist = read_file_names(file_list)
        processed_list = [] 
        for rawfilename in flist:        
            processed_file = process(rawfilename, runtype)
            processed_list.append(processed_file)            
        output_file = file_list + '_processed'
        with open(output_file, 'w') as file:
            for filename in processed_list:
                file.write(filename+'\n')
        print("Processed files saved to: ", output_file)
        
    except Exception as e:
        print("An error occurred while parsing arguments:", str(e))
        print("Usagee: python bin2h5df.py --runtype Saturation/TimeConstant --file_list file_list.txt")
        sys.exit(1)


if __name__ == '__main__':
    main()