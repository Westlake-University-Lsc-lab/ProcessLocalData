def determine_runtype(run_file):
    runtype = 'others'
    if 'calibration' in run_file:
        runtype = 'Saturation'
    elif '1p36v_0p68v' in run_file:
        runtype = 'TimeConstant'
    return runtype

def find_date(s):
    date = ''
    for sub_str in ['2024', '2025']:
        index = s.find(sub_str)         
        if index != -1:  
            shifted_index = index + 8
            if shifted_index < len(s):
                date_part = s[index : shifted_index]
                date = date_part
                break  
    # print(date)
    return date

def find_deta_t(s, runtype):
    if runtype == 'TimeConstant':
        delta_t = s.split('1p36v_0p68v_')[1].split('us_')[0]
    elif runtype == 'Saturation':
        delta_t = 5
    else:
        print("Error: Unknown run type")
        return None
    return delta_t

def find_trig_rate(s, runtype):
    if runtype == 'Saturation':
        trig_rate = s.split('calibration_')[1].split('Hz')[0]
        return trig_rate
    elif runtype == 'TimeConstant':
        trig_rate = s.split('us_')[1].split('Hz')[0]
        return trig_rate

def find_run_tag(s, runtype):
    run_tag = s.split('Hz_')[1].split('_run')[0]
    return run_tag

def find_voltage(s, runtype):
    date = find_date(s)
    if runtype == 'Saturation':
        voltage = s.find(date) +9
        if voltage != -1:  
            shifted_voltage = voltage + 4
            if shifted_voltage < len(s):
                voltage_part = s[voltage:shifted_voltage]
                voltage_value = float(voltage_part.replace('p', '.'))
                return voltage_value
    elif runtype == 'TimeConstant':        
        voltage = s.find(date) + 9
        if voltage != -1:  
            shifted_voltage = voltage + 3
            if shifted_voltage < len(s):
                voltage_part = s[voltage:shifted_voltage]
                voltage_value = float(voltage_part.replace('p', '.'))
                return voltage_value

def parse_run_info(rawfilename):
    runtype = determine_runtype(rawfilename)
    run_info = []
    date = find_date(rawfilename)
    volate = find_voltage(rawfilename, runtype)
    if runtype == 'TimeConstant':
        delta_t = find_deta_t(rawfilename, runtype)
    elif runtype == 'Saturation':
        delta_t = 5
    trig_rate = find_trig_rate(rawfilename, runtype)
    run_tag = find_run_tag(rawfilename, runtype)
    run_info = [{
        'date': date,
        'voltage': volate,
        'delta_t': delta_t,
        'trig_rate': trig_rate,
        'run_tag': run_tag,
        'file_tag': rawfilename[:-len('.bin')]
    }]
    return run_info
 
import os
def modify_file_names(target_date_str, modify_string, new_date_str):
    search_path = "/mnt/data/PMT/R8520_406/"
    if not os.access(search_path, os.F_OK):
        raise ValueError("Path not found or inaccessible: " + search_path)
    for root, dirs, files in os.walk(search_path):
        for filename in files:
            if target_date_str in filename:
                new_filename = filename.replace(modify_string, new_date_str)
                old_file_path = os.path.join(root, filename)
                new_file_path = os.path.join(root, new_filename)
                os.rename(old_file_path, new_file_path)
                print("--------------------")                
                print("New filename:", new_file_path)

