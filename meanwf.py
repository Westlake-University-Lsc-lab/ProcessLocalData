import pandas as pd
import numpy as np
import gc  # 导入垃圾回收模块
from decimal import Decimal, ROUND_HALF_UP

import analysis_data as ad
import process_data as prc
import plot_tools as pt
import runinfo as runinfo

import argparse
import os
import sys

def round_to_two(value):
    """将数值保留两位小数，处理None和NaN值"""
    if value is None or pd.isna(value):
        return value
    return float(Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

def main():
    try:
        parser = argparse.ArgumentParser(description='find the start time and 50 percent height of a combined mean waveform')
        parser.add_argument('--wftype', type=str, help='anode, dynode')
        parser.add_argument('--file_list', type=str, help='raw waveform files')
        
        args = parser.parse_args()
        
        if len(vars(args)) != 2:
            raise ValueError("Invalid number of arguments.")
            return 0
        
        wf_type = args.wftype
        file_list = args.file_list
        
        # 初始化列表来存储所有文件的信息
        all_wfinfo = []
        
        # 读取文件列表
        flist = ad.read_file_names(file_list)
        
        for f in flist:
            print(f'loading file:>>>> {f}')
            
            if not os.path.isfile(f):
                print(f"Warning: File {f} does not exist. Skipping.")
                continue
            ftag = runinfo.find_file_tag(f)
            print(f"file tag: {ftag}")
            runtype = runinfo.determine_runtype(ftag)
            print(f"runtype: {runtype}")
            voltage = runinfo.find_voltage(ftag, runtype)
            print(f"voltage: {voltage}")

            wfinfo = {}            
            try:
                df = pd.read_hdf(f)
                # 处理数据
                wf_arr = np.stack(df.Wave.to_numpy())
                wf_mean, wf_err = pt.mean_waveforms(wf_arr, 0, len(df.Wave[0]))
                
                wf_mean_filtered = prc.lowpass_filter(wf_mean, fs=250e6, cutoff=10e6, numtaps=10)
                baseline = df['Baseline'].mean()
                print('------------------')
                # print(f"baseline:{baseline}, mean:{wf_mean}")
                if wf_type == 'anode':
                    start, peak, t50, w50 = ad.find_pulse_width(wf_mean_filtered[10:], baseline=baseline, frac=0.5, polarity="negative")
                    _, _, t80, w80 = ad.find_pulse_width(wf_mean_filtered[10:], baseline=baseline, frac=0.8, polarity="negative")
                elif wf_type == 'dynode':
                    start, peak, t50, w50 = ad.find_pulse_width(wf_mean_filtered[10:], baseline=baseline, frac=0.5, polarity="positive")
                    _, _, t80, w80 = ad.find_pulse_width(wf_mean_filtered[10:], baseline=baseline, frac=0.8, polarity="positive")
                
                print(f"t50: {t50}, w50: {w50}")
                print(f"t50: {t80}, w50: {w80}")
                print('--------------------')
                # 填充信息字典
                # 将所有数值保留两位小数
                start = round_to_two(start)
                peak = round_to_two(peak)
                t50 = round_to_two(t50)
                w50 = round_to_two(w50)
                t80 = round_to_two(t80)
                w80 = round_to_two(w80)
                
                wfinfo = {
                    'voltage': voltage,
                    'baseline': baseline,
                    'start': start,
                    'peak': peak,
                    't50': t50,
                    'w50': w50*4E-3,
                    't80': t80,
                    'w80': w80*4E-3,
                    'wf_filtered': wf_mean_filtered,
                    'wf_mean': wf_mean,
                    'wf_err': wf_err,                      
                    'ftag': ftag,
                }
                
                # 将当前文件的信息添加到总列表中
                all_wfinfo.append(wfinfo)
                
                # 为每个文件单独保存
                path_save = f"/mnt/data/outnpy/{ftag}_mean.h5py"
                df0 = pd.DataFrame([wfinfo])  # 注意这里创建包含单个字典的列表
                prc.write_to_hdf5(df0, path_save)  
                print("save to >>>", path_save)
                
                # 显式删除大对象以释放内存
                del df, wf_arr, wf_mean, wf_err
                
                # 强制垃圾回收
                gc.collect()
                
            except Exception as e:
                print(f"Error processing file {f}: {e}")
                # 确保即使在出错时也尝试释放内存
                if 'df' in locals():
                    del df
                if 'wf_arr' in locals():
                    del wf_arr
                if 'wf_mean' in locals():
                    del wf_mean
                if 'wf_err' in locals():
                    del wf_err
                gc.collect()
                continue
                
        # 如果需要，也可以将所有文件的信息保存到一个总文件中
        if all_wfinfo:
            total_path_save = f"/mnt/data/outnpy/all_{wf_type}_files_combined.h5py"
            df_total = pd.DataFrame(all_wfinfo)
            prc.write_to_hdf5(df_total, total_path_save)
            print("Combined data saved to >>>", total_path_save)
            
            # 释放总数据框内存
            del df_total, all_wfinfo
            gc.collect()

    except Exception as e:
        print(f"Error: {e}")
        print('Usage: python meanwf.py --wftype [anode/dynode] --file_list flist')
        sys.exit(1) 

if __name__ == '__main__':
    main()