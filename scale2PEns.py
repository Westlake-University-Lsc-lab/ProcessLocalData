import numpy as np
import pandas as pd
import process_data
import constant
import argparse
import sys

Rft = constant.Rft
Rft_err= constant.Rft_err
GR_mu = constant.GR_mu
GR_err = constant.GR_err
    
def scale_S1_to_PEns(file, runtype='Saturation'):
    df = pd.read_hdf(file, key='winfo')
    PEns_filter = (df.Ch1_mu * Rft)/df.S1_width
    PEns_filter_err = (df.Ch1_mu * Rft)/df.S1_width * np.sqrt( (df.Ch1_sigma/df.Ch1_mu )**2 + (Rft_err/Rft)**2 + (1/df.S1_width)**2)
    PEns_dynode = (df.Ch2_mu * GR_mu) /df.S1_width
    PEns_dynode_err = (df.Ch2_mu * GR_mu )/df.S1_width * np.sqrt((df.Ch2_sigma/df.Ch2_mu)**2 + (GR_err/GR_mu)**2  + (1/df.S1_width)**2)
    PEns_anode = df.Ch0_mu /df.S1_width
    PEns_anode_err = df.Ch0_mu /df.S1_width * np.sqrt((df.Ch0_sigma/df.Ch0_mu) **2  + (1/df.S1_width)**2)
    df['PEns_filter'] = PEns_filter
    df['PEns_filter_err'] = PEns_filter_err
    df['PEns_dynode'] = PEns_dynode
    df['PEns_dynode_err'] = PEns_dynode_err
    df['PEns_anode'] = PEns_anode
    df['PEns_anode_err'] = PEns_anode_err
    
    file_tag = file.split('.h5py')[0]
    path = r'{}_{}_scaled.h5py'.format(file_tag,runtype)
    process_data.write_to_hdf5(df, path)
    return 0



def main():
    try:
        parser = argparse.ArgumentParser(description='scale area to PEns')
        parser.add_argument('--runtype', type=str, help='Saturation/TimeConstant')
        parser.add_argument('--file', type=str, help='file full path')
        args = parser.parse_args()
        
        if not args.runtype:
            print("Please provide runtype")
            print("Usagee: python sacle2PEns.py --runtype Saturation TimeConstant --file outnpy/*_single_gussain.h5py")
            return
        
        if args.runtype == 'Saturation':
            scale_S1_to_PEns(args.file, args.runtype)
        elif args.runtype == 'TimeConstant':
            scale_S1_to_PEns(args.file, args.runtype)
        else:
            print('Error: runtype not found')
            sys.exit(1)
    except Exception as e:    
        print(e)
        sys.exit(1)
        
if __name__ == '__main__':
    main()