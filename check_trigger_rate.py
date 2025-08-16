import pandas as pd
import daw_readout
#from matplotlib import pyplot as plt
import numpy as np
from tqdm import trange
from tqdm import tqdm
import sys

###################

####找出一个波形的起始点，最低点，结束点
def find_waveform_boundaries(waveform_data):
    # 找到最低点
    min_index = 0
    min_value = waveform_data[0]
    for i in range(1, len(waveform_data)):
        if waveform_data[i] < min_value:
            min_value = waveform_data[i]
            min_index = i
    
    # 从最低点向左遍历，找到回到基线的点，斜率为0
    start_index = min_index
    while start_index > 0 and (waveform_data[start_index] - waveform_data[start_index-1]) < 0  :
        start_index -= 1
    
    # 从最低点向右遍历，找到回到基线的点
    end_index = min_index
    while end_index < len(waveform_data)-1 and (waveform_data[end_index] - waveform_data[end_index+1]) < 0  :
        end_index += 1
    
    return start_index, min_index, end_index

###计算波形面积
def cal_area(
    waveform_data,
    st: 'int',
    ed: 'int',
    baseline: 'int'
):
    sum = np.sum( waveform_data[st: ed])
    area = baseline  * (ed - st) - sum  ### adc
    pe_fact  = (2./16384)*4.e-9/(50*1.6e-19)/1.e6  ## to PE
    return area*pe_fact

def cal_area_fix_width(
    waveform_data,
    st: 'int',
    fix_len: 'int',
    baseline : 'int',        
):
    sum = np.sum( waveform_data[st: st + fix_len])
    area = baseline * fix_len - sum
    pe_fact  = (2./16384)*4.e-9/(50*1.6e-19)/1.e6  ## to PE
    return area*pe_fact


argvs = sys.argv

#binary_file = '/mnt/data/PMT/R8520_406_LV2415/darkcurent_LV2415_20240628_20adc_darkbox_baseline_run0_raw_b0_seg0.bin'
binary_file =  sys.argv[1]   # '/mnt/data/PMT/R8520_406_LV2415/LED_LV2415_800V_20240620_run1_raw_b0_seg0.bin'
rawdata = daw_readout.DAWDemoWaveParser(binary_file)
winfo =[]
i=0
for wave in tqdm(rawdata) :
    ch = wave.Channel
    ttt = wave.Timestamp
    base = wave.Baseline
    st, minp, ed = find_waveform_boundaries(wave.Waveform)
    #ht = base - wave.Waveform[minp]
    #rst = minp  - st
    #flt = ed - minp
    #area = cal_area(wave.Waveform, st, ed, base)
    area_fixlen = cal_area_fix_width(wave.Waveform, st, 10, base)
    #area_fixrang = cal_led_area_fixlen(wave.Waveform, base)
    #pulse = wave.Waveform
    winfo.append({
        'EvtID': i,
        #'Ch':ch,
        'TTT':ttt,   ## Trigger time tag
        #'Baseline': base, 
        #'Hight': ht, 
        #'Width': ed -st,  ## pusle width
        #'Minp': minp,     ## min point
        #'Rst' : rst,      ## rise time of pulse    
        #'Flt' : flt,      ## full time of pulse  
        #'Area': area,
        'AreaFixlen': area_fixlen,
        #'AreaFixrange': area_fixrang,
        #'Wave': pulse,
    })
    i +=1
    print(f'total events:{} recording time len: {}  trigger rate: {}'.format(i,  np.max(ttt)*4.e-9, i/(np.max(ttt)*4.e-9) ) ))

