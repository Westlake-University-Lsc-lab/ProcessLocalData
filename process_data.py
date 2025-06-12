import pandas as pd
import numpy as np
import daw_readout
import time
import os

'''
find a pulse in segment, and return the start, end, peak index of the pulse
'''
def smooth_data(data, window_size=3):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def find_local_minima(data, window=3):
    minima = []
    for i in range(window, len(data)-window):
        if data[i] == np.min(data[i-window:i+window]):
            minima.append(i)
    return minima
def estimate_baseline(data, head_ratio=0.1, tail_ratio=0.1):
    head_samples = data[:int(len(data)*head_ratio)]
    tail_samples = data[-int(len(data)*tail_ratio):]
    return np.mean(np.concatenate([head_samples, tail_samples]))

def validate_pulse(
    start: int,
    end: int, 
    hight: int,
    threshold: float
) -> bool:
    """Validate if a pulse meets criteria.    
    Args:
        start (int): Pulse start index
        end (int): Pulse end index
        hight (int): Pulse height (amplitude)
        threshold (float): Minimum amplitude threshold
    
    Returns:
        bool: 
            - True if both conditions are met: 
                1. pulse_width >= 0 
                2. hight >= threshold
            - False otherwise
    """
    # Check if pulse duration is valid
    pulse_width = end - start + 1
    if pulse_width < 0:
        return False
    
    # Check if amplitude meets threshold
    if hight < threshold:
        return False
        
    # Both conditions satisfied
    return True

def pulse_hight(
    start: int, 
    end: int, 
    data: np.ndarray,
    baseline:int,
    ):
    """    
    Args:
        start (int): pulse start index
        end (int): pulse end index
        data (np.ndarray): pulse data
        baseline (int): pulse baseline
    
    Returns:
        float: pulse height
    """
    # calculate amplitude of pulse
    pulse_segment = data[start: end + 1]
    peak_value = np.max(pulse_segment) if abs(np.max(pulse_segment) - baseline) > abs(np.min(pulse_segment) - baseline) else np.min(pulse_segment)
    amplitude = abs(peak_value - baseline)
    return amplitude

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
        hight = pulse_hight(start, end, data, base)
        if validate_pulse(start, end, hight, 20.0) is True:
            pulses.append((start, end, min_idx, hight))
    return pulses
#######################################################################
'''
old version of pulse finding algorithm
'''
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
        
    return start_index, end_index,  min_index

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

###########################################################################
'''
function to calculate area of pulse
'''
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

############################################################################
'''
find main pulse in segment data
'''
def find_main_pulse(df, heigh_threshold=4000):
    """
    Main pulse detection algorithm
    """    
    area = df['Area']
    height = df['Hight']
    
    # main pulse detection
    if  height > heigh_threshold:  # PE and ADC thresholds
        return {
            'ttt': df['TTT'],
            'main_area': area,
            'main_height': height,
            'main_width': df['Width']
        }
    return None
'''
calculate the delay between main pulse and post events
'''
def process_all_segments(df, heigh_threshold):
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
        if main_pulse := find_main_pulse(row, heigh_threshold):
            # Save the previous main pulse
            if current_main:
                results.append(current_main)
            
            # initailize main pulse            
            current_main = {
                'main_ttt': main_pulse['ttt'],
                'main_area': main_pulse['main_area'],
                'main_height': main_pulse['main_height'],
                'main_width': main_pulse['main_width'],
                'post_events': []
            }
        
        # record post events
        elif current_main is not None:
            # if len(row['Area']) == 0:
            #     continue
            areas = row['Area'] if isinstance(row['Area'], (list, tuple)) else [row['Area']]
            heights = row['Hight'] if isinstance(row['Hight'], (list, tuple)) else [row['Hight']]

            for area, height in zip(areas, heights):
                time_interval = (row['TTT'] - current_main['main_ttt']) * 4  # 4ns
                current_main['post_events'].append({
                    'delay': time_interval,
                    'area': area,
                    'height': height
                })
    
    # Save the last main pulse
    if current_main:
        results.append(current_main)
    
    return pd.DataFrame(results)

###############################################################################
def read_txt_to_dataframe(file_path: str, runtype: str = 'Decay') -> pd.DataFrame:
    """
    reading TXT file to pandas DataFrame

    param：
    file_path (str) : txt file path

    return：
    pd.DataFrame : data including X and Y columns

    error:
    - FileNotFoundError: file not found
    - ValueError: axis identifier error or data type error
    - RuntimeError: data type error
    """
    X = []
    Y = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for _ in range(3):
                next(file)

            header_line = file.readline().strip()
            if header_line.replace('\t', ' ') != "X Y":
                raise ValueError(f"axis identifier error: should be 'X Y'，actual is '{header_line}'")
            
            for line_number, line in enumerate(file, start=5):  
                line = line.strip()
                if not line:
                    continue  # skip empty line
                if line == "</Trace>":  # end of trace
                    print(f"checking end of trace '</Trace>'，stop reading.")
                    break
                parts = line.split()                
                if len(parts) != 2:
                    raise RuntimeError(f"number {line_number} line data type error: shoulde be 2，actual is {len(parts)}")

                try:
                    if runtype == 'Spectrum':
                        x_val = int(parts[0])
                        y_val = float(parts[1])
                    elif runtype == 'Decay':
                        x_val = float(parts[0])
                        y_val = int(parts[1])
                    
                    else:
                        raise ValueError(f"unkown runtype: {runtype}")
                    # print(x_val, y_val)
                except ValueError as e:
                    raise RuntimeError(f"number {line_number} line data type error: {e}")
                X.append(x_val)
                Y.append(y_val)

    except FileNotFoundError:
        raise FileNotFoundError(f"file {file_path} not found") from None

    return pd.DataFrame({'X': X, 'Y': Y})

##########################################################################################
def selection_main_pulse_0(df):
    # 处理Ch==0的通道并按TTT排序
    df_ch0 = df[df['Ch'] == 0].sort_values('TTT').reset_index(drop=True)
    if df_ch0.empty:
        return []
    
    # 定义主脉冲的周期（1秒对应的TTT单位）
    pulse_period = 250_000_000  # 1秒 = 250000000 * 4ns
    # error_margin = 1  # 允许的误差范围（±2个TTT单位，即±8ns）
    
    # 存储所有可能的主脉冲序列
    candidate_sequences = []
    
    # 遍历所有数据点，寻找可能的主脉冲起点
    # for start_idx in range(0, 10000):
    for start_idx in range(len(df_ch0)):
        current_sequence = [start_idx]
        current_index = start_idx
        
        while True:
            current_ttt = df_ch0.loc[current_index, 'TTT']
            expected_next_ttt = current_ttt + pulse_period
            
            # 查找符合条件的候选
            mask = (df_ch0['TTT'] >= expected_next_ttt ) & \
                   (df_ch0['TTT'] <= expected_next_ttt )
            candidates = df_ch0.index[mask].tolist()
            candidates = [idx for idx in candidates if idx > current_index]
            
            if not candidates:
                break  # 无后续候选，终止循环
            
            # 选择最接近的候选（若有多个，选最小的索引）
            next_index = min(candidates)
            current_sequence.append(next_index)
            current_index = next_index
        
        # 如果找到的序列长度大于1，则作为候选序列
        if len(current_sequence) > 1:
            candidate_sequences.append(current_sequence)
    
    # 如果没有找到任何候选序列，返回空列表
    if not candidate_sequences:
        return []
    
    # 选择最长的候选序列作为主脉冲序列
    main_pulses = max(candidate_sequences, key=len)
    
    # 构建输出结果
    output = []
    for i in range(len(main_pulses)):
        main_idx = main_pulses[i]
        main_data = df_ch0.loc[main_idx]
        main_ttt = main_data['TTT']
        
        post_events = []
        if i < len(main_pulses) - 1:
            # 收集当前main_pulse到下一个main_pulse之间的事件
            next_main_idx = main_pulses[i + 1]
            post_indices = range(main_idx + 1, next_main_idx)
        else:
            post_indices = []  # 最后一个main_pulse不处理后续事件
        
        for idx in post_indices:
            event = df_ch0.loc[idx]
            delay = (event['TTT'] - main_ttt) * 4  # 转换为ns
            post_events.append({
                'delay': delay,
                'area': event['Area'],
                'height': event['Hight'],
                'width': event['Width']
            })
        
        output.append({
            'main_ttt': main_ttt,
            'main_area': main_data['Area'],
            'main_height': main_data['Hight'],
            'main_width': main_data['Width'],
            'post_events': post_events
        })
    
    return pd.DataFrame(output)

# import pandas as pd
# import numpy as np
# import pandas as pd
# import numpy as np

def selection_main_pulse_1(df):
    """
    寻找 LED 主脉冲及其后续事件。
    
    参数:
        df (pd.DataFrame): 输入数据，包含 ['Ch', 'TTT', 'Area'] 列。
    
    返回:
        list: 包含主脉冲及其后续事件的列表。
    """
    # 1. 预处理：筛选通道并排序
    df_ch0 = df[df['Ch'] == 0].sort_values('TTT').reset_index(drop=True)
    if df_ch0.empty:
        return []
    
    # 2. 定义常量
    TTT_UNIT = 4  # 1 TTT 单位 = 4ns
    PULSE_PERIOD = 250_000_000  # 1秒对应的 TTT 单位 (1e9 / 4 = 250e6)
    WINDOW_SIZE = 60 * PULSE_PERIOD  # 1分钟时间窗口（60秒）
    
    # 3. 将 TTT 转换为 numpy 数组以便快速操作
    ttt_array = df_ch0['TTT'].values
    area_array = df_ch0['Area'].values
    
    # 4. 寻找主脉冲序列
    main_pulses = []
    for i in range(len(ttt_array)):
        # 检查当前时间窗口是否在 1 分钟内
        if i > 0 and ttt_array[i] - ttt_array[0] > WINDOW_SIZE:
            break
        
        # 尝试以当前点为起点，寻找符合周期的主脉冲序列
        current_sequence = [i]
        current_ttt = ttt_array[i]
        
        # 向后查找符合周期的主脉冲
        for j in range(i + 1, len(ttt_array)):
            expected_ttt = current_ttt + PULSE_PERIOD
            if abs(ttt_array[j] - expected_ttt) <= 2:  # 允许 ±8ns 误差
                current_sequence.append(j)
                current_ttt = ttt_array[j]
        
        # 如果找到符合条件的主脉冲序列，保存并退出
        if len(current_sequence) >= 2:  # 至少有两个脉冲
            main_pulses = current_sequence
            break
    
    # 5. 如果没有找到主脉冲序列，返回空列表
    if not main_pulses:
        return []
    
    # 6. 标记主脉冲和后续事件
    output = []
    for i in range(len(main_pulses)):
        main_idx = main_pulses[i]
        main_ttt = ttt_array[main_idx]
        main_area = area_array[main_idx]
        
        # 确定后续事件范围
        if i < len(main_pulses) - 1:
            next_main_idx = main_pulses[i + 1]
            post_indices = range(main_idx + 1, next_main_idx)
        else:
            post_indices = range(main_idx + 1, len(ttt_array))
        
        # 收集后续事件
        post_events = []
        for idx in post_indices:
            delay = (ttt_array[idx] - main_ttt) * TTT_UNIT  # 转换为 ns
            post_events.append({
                'delay': int(delay),
                'area': area_array[idx]
            })
        
        # 添加到输出结果
        output.append({
            'main_ttt': int(main_ttt),
            'main_area': main_area,
            'post_events': post_events
        })
    
    return pd.DataFrame(output)


def find_periodic_elements(arr, step, known_value):
    """
    args:
        arr (list): 输入的数组。
        step (int): 步长（周期）。
        known_value: 已知的数组中的一个元素值。
    
    return:
        list: 符合步长条件的元素列表。
    """
    # 构建哈希表：值 -> 索引
    value_to_indices = {}
    for idx, value in enumerate(arr):
        print(idx, value)
        if value not in value_to_indices:
            value_to_indices[value] = []
        value_to_indices[value].append(idx)
    
    # 如果已知值不在数组中，返回空列表
    if known_value not in value_to_indices:
        return []
    
    result = []
    
    # 从已知值开始，向前和向后查找符合步长的元素
    for start_idx in value_to_indices[known_value]:
        # 向前查找
        print('to left---------------')        
        current_idx = start_idx
        while current_idx >= 0:
            result.append(arr[current_idx])
            current_idx -= step
        print('to right---------------')
        # 向后查找
        current_idx = start_idx + step
        while current_idx < len(arr):
            result.append(arr[current_idx])
            current_idx += step
    
    # 去重并返回结果
    return list(set(result))

import numpy as np

def find_periodic_elements_optimized(arr, step, known_value):
    """
    优化版：直接操作 numpy.ndarray，避免转换为 list。
    """
    # 如果输入不是 numpy.ndarray，转换为 numpy.ndarray
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    
    # 找到已知值的所有索引
    known_indices = np.where(arr == known_value)[0]
    
    # 如果已知值不在数组中，返回空列表
    if len(known_indices) == 0:
        return []
    
    # 初始化结果集合
    result = set()
    
    # 从已知值开始，向前和向后查找符合步长的元素
    for start_idx in known_indices:
        # 向前查找
        current_idx = start_idx
        while current_idx >= 0:
            result.add(arr[current_idx])
            current_idx -= step
        
        # 向后查找
        current_idx = start_idx + step
        while current_idx < len(arr):
            result.add(arr[current_idx])
            current_idx += step
    
    # 返回结果列表
    return list(result)


def selection_main_pulse(df):
    # 预处理：筛选通道并排序
    df_ch0 = df[df['Ch'] == 0].sort_values('TTT').reset_index(drop=True)
    if df_ch0.empty:
        return []
    
    # 定义常量
    TTT_UNIT = 4  # 1 TTT单位=4ns
    PULSE_PERIOD = 250_000_000  # 1秒对应的TTT单位 (1e9 / 4 = 250e6)
    
    # 获取最大Area的脉冲作为起始点
    max_area_idx = df_ch0['Area'].idxmax()
    main_ttt = df_ch0.at[max_area_idx, 'TTT']
    
    print(r'main_pulse area:{}, ttt:{}'.format(max_area_idx, main_ttt))
    # 构建TTT数组用于快速搜索
    ttt_array = df_ch0['TTT'].values
    # indices = np.arange(len(ttt_array))
    main_ttts = find_periodic_elements(ttt_array, PULSE_PERIOD, main_ttt)
    
    print(r'found {} main_pulse ttt(s)'.format(len(main_ttts)))
    print(r'main_pulse rate:{}Hz'.format(len(main_ttts) /  ((df.TTT.max() - df.TTT.min()) *4.E-9 )))
                      
                       
    # 验证脉冲间隔误差
    valid_main_ttts = []
    for i in range(1, len(main_ttts)):
        delta = main_ttts[i] - main_ttts[i-1]
        if abs(delta - PULSE_PERIOD) == 0:
            valid_main_ttts.extend([main_ttts[i-1], main_ttts[i]])
        else:
            print(r'Warning: invalid pulse interval: {}ns'.format(delta))            
    valid_main_ttts = sorted(list(set(valid_main_ttts)))
    
    # 构建输出结构
    output = []
    ttt_to_index = {ttt: idx for idx, ttt in enumerate(ttt_array)}
    for i in range(len(valid_main_ttts)):
        main_ttt = valid_main_ttts[i]
        main_idx = ttt_to_index[main_ttt]
        main_data = df_ch0.iloc[main_idx]
        
        # 确定post_events范围
        next_main_ttt = valid_main_ttts[i+1] if i+1 < len(valid_main_ttts) else None
        if next_main_ttt:
            end_idx = ttt_to_index[next_main_ttt]
            post_indices = range(main_idx+1, end_idx)
        else:
            post_indices = []
        
        # 收集post_events
        post_events = []
        for idx in post_indices:
            event = df_ch0.iloc[idx]
            delay = (event['TTT'] - main_ttt) * TTT_UNIT
            post_events.append({
                'delay': int(delay),
                'area': event['Area'],
                'height': event['Hight'],
                'width': event['Width']
            })
        
        output.append({
            'main_ttt': int(main_ttt),
            'main_area': main_data['Area'],
            'main_height': main_data['Hight'],
            'main_width': main_data['Width'],
            'post_events': post_events
        })
    result =pd.DataFrame(output)
    return result