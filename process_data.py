import pandas as pd
import numpy as np
import daw_readout
import time
import os

def smooth_data(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def find_local_minima(data, window=10):
    minima = []
    for i in range(window, len(data)-window):
        if data[i] == np.min(data[i-window:i+window]):
            minima.append(i)
    return minima
def estimate_baseline(data, head_ratio=0.1, tail_ratio=0.1):
    head_samples = data[:int(len(data)*head_ratio)]
    tail_samples = data[-int(len(data)*tail_ratio):]
    return np.mean(np.concatenate([head_samples, tail_samples]))

def validate_pulse(start: int, end: int, data: np.ndarray, threshold: float) -> bool:
    """    
    Args:
        start (int): pulse start index
        end (int): pulse end index
        data (np.ndarray): pulse data
        threshold (float): pulse amplitude threshold
    
    Returns:
        bool: True (valid) / False(invalid)
    """
    pulse_width = end - start + 1
    if pulse_width <= 3:
        return False
    
    # calculate mean value using the pre and post 10 samples of pulse as baseline
    pre_pulse = data[max(0, start - 10): start]
    post_pulse = data[end + 1: min(len(data), end + 10)]
    baseline = np.mean(np.concatenate([pre_pulse, post_pulse])) if len(pre_pulse) + len(post_pulse) > 0 else np.median(data)
    
    # calculate amplitude of pulse
    pulse_segment = data[start: end + 1]
    peak_value = np.max(pulse_segment) if abs(np.max(pulse_segment) - baseline) > abs(np.min(pulse_segment) - baseline) else np.min(pulse_segment)
    amplitude = abs(peak_value - baseline)
    
    return amplitude >= threshold

def improved_pulse_index(data, min_idx, baseline):   
    threshold_ratio = 0.1  
    gradient_threshold = 0.5  
    # baseline = estimate_baseline(data)   
    start_index = min_idx
    while start_index > 0:
        delta = data[start_index] - baseline
        grad = data[start_index] - data[start_index-1]
        if delta > threshold_ratio * (baseline - min_idx) or grad > gradient_threshold:
            break
        start_index -= 1         
    end_index = min_idx
    while end_index < len(data)-1 :
        delta = data[end_index] - baseline
        grad = data[end_index] - data[end_index+1]
        if delta > threshold_ratio * (baseline - min_idx) or grad > gradient_threshold:
            break
        end_index += 1 
    start_index = max(0, start_index)
    end_index = min(len(data)-1, end_index)    
    return start_index,  end_index

    
def detect_all_pulses(data, base):
    pulses = []
    local_minima = find_local_minima(data)
    for min_idx in local_minima:
        start,  end = improved_pulse_index(data, min_idx, base)
        if validate_pulse(start, end, data, 20.0): 
            pulses.append((start, end, min_idx))
    return pulses

### read data and clculate start, min, end index ###
def pulse_index(waveform_data):
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
        
    return start_index, min_index, end_index

def find_waveform_intersections(waveform, baseline, negative_pulse=True, percent=0.1):
    if negative_pulse:
        waveform_below_baseline = waveform[waveform < baseline]
        if len(waveform_below_baseline) == 0:
            return None, None, None
        peak_value = np.min(waveform_below_baseline)
        peak_index = np.argmin(waveform)
        height = peak_value - baseline
        threshold = baseline + percent * height
    crossings = np.where(np.diff(np.sign(waveform - threshold)) != 0)[0]
    if len(crossings) == 0:
        return peak_index, None, None
    min_index = crossings[0]
    max_index = crossings[-1]
    return peak_index, min_index, max_index


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

def cal_rms(data):
    import math
    squared_sum = sum(element**2 for element in data)
    average_squared = squared_sum / len(data)
    rms = math.sqrt(average_squared)
    return rms


def find_main_pulse(segment):
    """
    Main pulse detection algorithm
    """
    # Check if the segment has only one pulse
    if len(segment['Area']) != 1 or len(segment['hight']) != 1:
        return None
    
    area = segment['Area'][0]
    height = segment['hight'][0]
    
    # main pulse detection
    if  height > 1000:  # PE and ADC thresholds
        return {
            'ttt': segment['TTT'],
            'main_area': area,
            'main_height': height,
            'Wave': segment['Wave']
        }
    return None

def process_all_segments(df):
    """
    Process all segments in the dataframe
    """
    results = []
    current_main = None
    
    # Sort the dataframe by TTT
    df = df.sort_values('TTT')    
    for idx, row in df.iterrows():
        if row['Ch'] != 0:
            continue
        if main_pulse := find_main_pulse(row):
            # Save the previous main pulse
            if current_main:
                results.append(current_main)
            
            # initailize main pulse            
            current_main = {
                'main_ttt': main_pulse['ttt'],
                'main_area': main_pulse['main_area'],
                'main_height': main_pulse['main_height'],
                'post_events': []
            }
        
        # record post events
        elif current_main is not None:
            if len(row['Area']) == 0:
                continue
            
            for area, height in zip(row['Area'], row['hight']):
                time_interval = row['TTT'] - current_main['main_ttt'] * 4.e-9 # 4ns                
                current_main['post_events'].append({
                    'delay': time_interval,
                    'area': area,
                    'height': height
                })
    
    # Save the last main pulse
    if current_main:
        results.append(current_main)
    
    return pd.DataFrame(results)
