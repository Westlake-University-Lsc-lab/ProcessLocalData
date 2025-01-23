import pandas as pd
import numpy as np
import sys
import argparse
import process_data

def group_by_voltage(file):
    df = pd.read_hdf(file, key='winfo')
    grouped_lists = []
    for voltage_value in df['Voltage'].unique():   
        filtered_df = df[df['Voltage'] == voltage_value] 
        group_dict = {'Voltage': voltage_value, 'runinfo': filtered_df}   
        grouped_lists.append(group_dict)
    return grouped_lists

def calculate_R2ref(df, column_name):
    condition = df.Delta_t == '1000'
    assert isinstance(df, pd.DataFrame)
    if not df.empty:  
        ref_mu = df[column_name][condition].values[0]  
        ref_mu_err = df[f'{column_name}_err'][condition].values[0]  
        if ref_mu != 0:  
            R2ref = df[column_name] / ref_mu
            R2ref_err = R2ref * np.sqrt((df[f'{column_name}_err'] / df[column_name])**2 + (ref_mu_err / ref_mu)**2)
            df[f'R2ref_{column_name}'] = R2ref
            df[f'R2ref_{column_name}_err'] = R2ref_err
            return df
        else:  
            print('ref_mu is zero, cannot calculate R2ref')
            sys.exit(0)                      
    else:
        print('no data for this voltage value')
        sys.exit(0)

def process_group_dict(group_lists):
    for group in group_lists:
        df = group['runinfo']
        if isinstance(df, pd.DataFrame):     
            calculate_R2ref(df, 'PEns_anode')
    return group_lists 

def main():
    try:
        parser = argparse.ArgumentParser(description='Calculate R2ref in TimeConstant runs')
        parser.add_argument('--file', type=str, help='sacled file')
        args = parser.parse_args()       
        if not args.file:
            print("Please provide file")
            print("Usagee: python sacle2PEns.py --file outnpy/*single_gussain_TimeConstant_scaled.h5py")
            return
                
        grouped_lists = group_by_voltage(args.file)
        grouped_lists =process_group_dict(grouped_lists)
        runinfo = []
        for g in grouped_lists:
           for index, row in g['runinfo'].iterrows():
               runinfo.append({
                   'Voltage': row['Voltage'],
                   'R2ref_anode': row['R2ref_PEns_anode'],
                   'R2ref_anode_err': row['R2ref_PEns_anode_err'],
                   'Delta_t': row['Delta_t']
                   })
        df = pd.DataFrame(runinfo)
        file_tag = args.file.split('.h5py')[0]
        print(file_tag)
        path_save = r'{}_R2ref.h5py'.format(file_tag)
        process_data.write_to_hdf5(df, path_save)  
        print(path_save)
        return  path_save
        
    except Exception as e:
        print("An error occurred while parsing arguments:", str(e))
        print('Usage: python CalR2ref.py --file TimeConstant_scaled.h5py')
        sys.exit(1)    
            
if __name__ == '__main__':
    main()