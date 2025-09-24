import pandas as pd
import numpy as np
import fit_package
import process_data
import argparse
import sys

def fit_spectrum(flist):
    run_info = []
    f_tag = ''
    with open(flist, 'r') as list:
        for line in list:       
            file = line.rstrip('\n')
            f_tag = fit_package.ftag(file)
            df = pd.read_hdf(file, key='winfo')                
            ch0_mu,ch0_sigma = fit_package.fit_single_channel(df, 0, f_tag)               
            ch1_mu,ch1_sigma = fit_package.fit_single_channel(df, 1, f_tag)
            ch2_mu,ch2_sigma = fit_package.fit_single_channel(df, 2, f_tag)
            # ch5_mu,ch5_sigma = fit_package.fit_single_channel(df, 5, f_tag)
            run_info.append({               
                'S2_width': df.S2_width.values[0],
                'S1_width': df.S1_width.values[0],
                'Delta_t': df.Delta_t.values[0],
                'Voltage': df.Voltage.values[0],
                'RunTag': df.RunTag.values[0],
                'ftag': f_tag,   
                'ch0_mu': ch0_mu,
                'ch1_mu': ch1_mu,
                'ch2_mu': ch2_mu,
                # 'ch5_mu': ch5_mu,
                'ch0_sigma' : ch0_sigma,
                'ch1_sigma' : ch1_sigma,
                'ch2_sigma' : ch2_sigma,         
                # 'ch5_sigma' : ch5_sigma,         
            })        
    return run_info, f_tag

def main():
   
    try:
        parser = argparse.ArgumentParser(description='fit spectrum with gaussian model')
        parser.add_argument('--file_list', type=str, help='file list')
        args = parser.parse_args()
 
        if not args.file_list:
            print("Please provide file_list")
            print("Usagee: python guassainfitspectrum.py --file_list file_list.txt")
            return
       
        file_list = args.file_list
        run_info, f_tag = fit_spectrum(file_list)        
        df_new = pd.DataFrame(run_info)
        path = r'/mnt/data/outnpy/{}_single_gussain.h5py'.format(f_tag)
        process_data.write_to_hdf5(df_new, path)
    
    except Exception as e:
        print(e)
        print('Please provide all required arguments')
        print("Usagee: python guassainfitspectrum.py --file_list file_list.txt")
        sys.exit(1)


if __name__ == '__main__':
    main()
