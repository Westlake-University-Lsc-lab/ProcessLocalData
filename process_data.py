import pandas as pd
import numpy as np
import daw_readout


### read data and clculate start, min, end index ###
def pusle_index(waveform_data):
    ### find min index ###
    min_index = 0
    min_value = waveform_data[0]
    for i in range(1, len(waveform_data)-1):
        if waveform_data[i] < min_value:
            min_value = waveform_data[i]
            min_index = i
    
    ### searching from min index position to left till back to baseline 
    start_index = min_index
    while start_index > 0 and (waveform_data[start_index] - waveform_data[start_index-1]) < 0  :
        start_index -= 1    
    ### searching form min index position to right till back to baseline
    end_index = min_index
    if (waveform_data[min_index] == waveform_data[min_index+1]):
        end_index = min_index+1    
    while end_index < len(waveform_data)-1 and (waveform_data[end_index] - waveform_data[end_index+1]) < 0  :
        end_index += 1    
    return start_index, min_index, end_index

### calculate area of puse with dynamic range ###
def pusle_area(
    waveform_data,
    st: 'int',
    ed: 'int',
    baseline: 'int'
):
    sum = np.sum( waveform_data[st: ed])
    area = baseline  * (ed - st) - sum  ### adc
    pe_fact  = (2./16384)*4.e-9/(50*1.6e-19)/1.e6  ## to PE
    return area*pe_fact

### calculate area of pulse with fixed width ###
def pulse_area_fix_len(
    waveform_data,
    st: 'int',
    fix_len: 'int',
    baseline : 'int',        
):
    sum = np.sum( waveform_data[st: st + fix_len])
    area = baseline * fix_len - sum
    pe_fact  = (2./16384)*4.e-9/(50*1.6e-19)/1.e6  ## to PE
    return area*pe_fact
    