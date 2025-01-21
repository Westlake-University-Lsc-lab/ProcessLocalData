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

def scale_S2_to_PEns(file):
    df = pd.read_hdf(file, key='winfo')
    ''' 
        Constant parameters:
    '''
    Rft = constant.Rft
    Rft_err= constant.Rft_err
    GR_mu = constant.GR_mu
    GR_err = constant.GR_err
    ref_mu = df.Ch0_s2_mu[df.Delta_t == 1000]
    ref_err = df.Ch0_s2_sigma[df.Delta_t == 1000]
    ref_width = df.S2_width[df.Delta_t == 1000] 
    ref_width = df.S2_width[df.Delta_t == 1000] ### unit in ns
    ''' 
        Calculate new Parameters:
    '''
    PEns_ref = float(ref_mu.iloc[0] / ref_width.iloc[0])
    PEns_ref_err = float(ref_mu.iloc[0] / ref_width.iloc[0]) *np.sqrt((ref_err.iloc[0]/ref_mu.iloc[0])**2 + (2/ref_width.iloc[0])**2 )
    
    S2_PEns_filter = (df.Ch1_s2_mu * Rft)/df.S2_width
    S2_PEns_filter_err =(df.Ch1_s2_mu * Rft)/df.S2_width * np.sqrt((df.Ch1_s2_sigma/df.Ch1_s2_mu)**2 + (Rft_err/Rft)**2  + (2/df.S2_width)**2)
    S2_PEns_anode = df.Ch0_s2_mu /df.S2_width
    S2_PEns_anode_err = df.Ch0_s2_mu /df.S2_width * np.sqrt((df.Ch0_s2_sigma/df.Ch0_s2_mu) **2  + (2/df.S2_width)**2)
    S2_PEns_dynode = df.Ch2_s2_mu * GR_mu /df.S2_width
    S2_PEns_dynode_err =df.Ch2_s2_mu * GR_mu /df.S2_width * np.sqrt((df.Ch2_s2_sigma/df.Ch2_s2_mu)**2 + (GR_err/GR_mu)**2  + (2/df.S2_width)**2)
    R2ref_filter = S2_PEns_filter /PEns_ref
    R2ref_filter_err = S2_PEns_filter /ref_mu * np.sqrt((PEns_ref_err/PEns_ref) **2  + (S2_PEns_filter_err/S2_PEns_filter) **2)
    R2ref= S2_PEns_anode/PEns_ref
    R2ref_err = S2_PEns_anode /PEns_ref * np.sqrt((PEns_ref_err/PEns_ref) **2  + (S2_PEns_anode_err/S2_PEns_anode) **2)
    ''' 
        save to dataframe 
    '''
    
    df['R2ref'] = R2ref
    df['R2ref_err'] = R2ref_err
    df['R2ref_filter'] = R2ref_filter
    df['R2ref_filter_err']= R2ref_filter_err
    df['PEns_anode'] = S2_PEns_anode
    df['PEns_anode_err'] = S2_PEns_anode_err
    df['PEns_filter'] = S2_PEns_filter
    df['PEns_fliter_err'] = S2_PEns_filter_err
    df['PEns_dynode'] = S2_PEns_dynode
    df['PEns_dynode_err'] = S2_PEns_dynode_err
    
    #### save to hdf5 file    
    file_tag = file.split('.h5py')[0]
    path = r'{}_scaled.h5py'.format(file_tag)
    # print('saving to {}'.format(path))
    process_data.write_to_hdf5(df, path)
    
    return 0
    
    
    
def scale_S1_to_PEns(file):
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
    path = r'{}_scaled.h5py'.format(file_tag)
    # print('saving to {}'.format(path))
    process_data.write_to_hdf5(df, path)
    return 0



def main():
    try:
        parser = argparse.ArgumentParser(description='scale area to PEns')
        parser.add_argument('--runtype', type=str, help='Saturation/TimeConstant')
        parser.add_argument('--file', type=str, help='file full path')
        parser.add_argument('--file_list', type=str, help='file list name need to scale')
        args = parser.parse_args()
        
        if not args.runtype:
            print("Please provide runtype")
            print("Usagee: python sacle2PEns.py --runtype Saturation --file outnpy/*_single_gussain.h5py  or ")
            print("Usagee: python sacle2PEns.py --runtype TimeConstant --file_list file_list")
            return
        
        if args.runtype == 'Saturation':
            scale_S1_to_PEns(args.file)
        elif args.runtype == 'TimeConstant':
            with open(args.file_list, 'r') as list:
                for line in list:       
                    file = line.rstrip('\n')
                    scale_S2_to_PEns(file)
        else:
            print('Error: runtype not found')
            sys.exit(1)
    except Exception as e:    
        print(e)
        sys.exit(1)
        
if __name__ == '__main__':
    main()