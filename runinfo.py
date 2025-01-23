def determine_runtype(run_file):
    runtype = 'others'
    if 'calibration' in run_file:
        runtype = 'Saturation'
    elif '1p36v_0p68v' in run_file:
        runtype = 'TimeConstant'
    elif 'LongS2' in run_file:
        runtype = 'LongS2'
    else:
        runtype = 'others'
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

def find_file_tag(s, runtype):
    file_tag = s.split('/mnt/data/PMT/R8520_406/')[1].split('.bin')[0]
    return file_tag

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

def parse_run_info(rawfilename, runtype):
    # runtype = determine_runtype(rawfilename)
    run_info = []
    date = find_date(rawfilename)
    volate = find_voltage(rawfilename, runtype)
    if runtype == 'TimeConstant':
        delta_t = find_deta_t(rawfilename, runtype)
    elif runtype == 'Saturation' or runtype == 'LongS2':
        delta_t = 5
    trig_rate = find_trig_rate(rawfilename, runtype)
    run_tag = find_run_tag(rawfilename, runtype)
    file_tag = find_file_tag(rawfilename, runtype)
    run_info = [{
        'date': date,
        'voltage': volate,
        'delta_t': delta_t,
        'trig_rate': trig_rate,
        'run_tag': run_tag,
        'file_tag': file_tag
    }]
    return run_info
 
