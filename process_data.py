import pandas as pd
import numpy as np
import daw_readout
import time
import os


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
        start_index -=1         
     
    ### searching form min index position to right till back to baseline
    end_index = min_index
    if (waveform_data[min_index] == waveform_data[min_index+1]):
        end_index = min_index+1    
    while end_index < len(waveform_data)-1 and (waveform_data[end_index] - waveform_data[end_index+1]) < 0  :
        end_index +=1
        
    #### refind end_index direct to right if there is a peak in the right hand side of end_index
    #if  end_index < min_index + 20 and waveform_data[end_index +1 ] <= waveform_data[end_index] :
    #    mind_index_1 = end_index
    #    min_value_1 = waveform_data[end_index]
    #    for i in range(mind_index_1 , min_index + 20):
    #        if waveform_data[i] < min_value_1:
    #            min_value_1 = waveform_data[i]
    #            mind_index_1 = i
    #    end_index = mind_index_1
    #    while end_index < min_index + 20 and (waveform_data[end_index] - waveform_data[end_index+1]) < 0  :
    #        end_index +=1
    
    #### refind start_index direct to left if there is a peak in the left hand side of start_index
    #if start_index > min_index - 20 and waveform_data[start_index -1 ] <= waveform_data[start_index] :
    #    mind_index_1 = start_index
    #    min_value_1 = waveform_data[start_index]
    #    for i in range(min_index - 20, mind_index_1 -1):
    #        if waveform_data[i] < min_value_1:
    #            min_value_1 = waveform_data[i]
    #            mind_index_1 = i
    #    start_index = mind_index_1
    #    while start_index >  min_index - 20 and (waveform_data[start_index] - waveform_data[start_index-1]) < 0  :
    #        start_index -=1
  
    return start_index, min_index, end_index

### calculate area of puse with dynamic range ###
pe_fact  = (2./16384)*4.e-9/(50*1.6e-19)/1.e6  ## to PE
def pulse_area(
    waveform_data,
    st: 'int',
    ed: 'int',
    baseline: 'int'
):
    sum = np.sum( waveform_data[st: ed])
    area = baseline  * (ed - st) - sum  ### adc
    return area
    #return area*pe_fact

### calculate area of pulse with fixed width ###
def pulse_area_fix_len(
    waveform_data,
    minp: 'int',
    fix_len: 'int',
    baseline : 'int',        
):
    sum = np.sum( waveform_data[minp-fix_len : minp+fix_len])
    area = baseline * fix_len*2 - sum
    #pe_fact  = (2./16384)*4.e-9/(50*1.6e-19)/1.e6  ## to PE
    return area

def pulse_area_fix_range(
    waveform_data,
    left: 'int',
    right: 'int',
    baseline : 'int',        
):
    sum = np.sum( waveform_data[left: right])
    area = baseline * (right - left) - sum
    return area

def write_to_hdf5(df, filename):
    start_time = time.time()
    df.to_hdf(filename, key='winfo', mode='w', complib='blosc:blosclz', complevel=9)  
    write_time = time.time() - start_time
    file_size = os.path.getsize(filename)
    print(r"h5 Write Time: {:.2f} s ".format(write_time))
    print(r"h5 File Size: {:.2f} MB".format( file_size/(1024*1024)) )
    print("Save to {}".format(filename))
    return write_time,  file_size
