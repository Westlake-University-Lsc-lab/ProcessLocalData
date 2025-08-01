def determine_runtype(run_file):
    runtype = 'others'
    if 'calibration' in run_file:
        runtype = 'Saturation'
    elif '1p36v_0p68v' in run_file:
        runtype = 'TimeConstant'
    elif 'Calibration_1kHz' in run_file:
        runtype = 'Calibration'    
    elif 'LongS2' in run_file:
        runtype = 'LongS2'
    elif 'DecayConstant' in run_file:
        runtype = 'DecayConstant'
    elif 'darkrate' in run_file:
        runtype = 'DarkRate'
    elif 'self' in run_file:
        runtype = 'DarkRate'
    else:
        runtype = 'others'
    print(runtype)
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
    print(date)
    return date

def find_deta_t(s, runtype):
    if runtype == 'TimeConstant':
        delta_t = s.split('1p36v_0p68v_')[1].split('us_')[0]
    else:
        delta_t = None
    return delta_t

def find_trig_rate(s, runtype):
    if runtype == 'Saturation':
        trig_rate = s.split('calibration_')[1].split('Hz')[0]
        return trig_rate
    elif runtype == 'TimeConstant':
        trig_rate = s.split('us_')[1].split('Hz')[0]
        return trig_rate
    elif runtype == 'Calibration':
        trig_rate = s.split('Calibration_')[1].split('Hz')[0]
        return trig_rate
    else:
        trig_rate = None
        return trig_rate
        

def find_run_tag(s, runtype):
    run_tag = s.split('Hz_')[1].split('_run')[0]
    return run_tag

# def find_file_tag(s, runtype):
#     file_tag = s.split('/mnt/data/PMT/R8520_406/')[1].split('.bin')[0]
#     return file_tag
##----
def find_file_tag(s):
    prefixes = ['/mnt/data/PMT/R8520_406/', '/mnt/data/TPC/']
    for prefix in prefixes:
        if prefix in s:
            try:
                file_tag = s.split(prefix)[1].split('.bin')[0]
                return file_tag
            except IndexError:
                # 如果格式不符合预期，返回None或抛异常
                return None
    return None
###---

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
    elif runtype == 'Calibration':
        voltage = s.find(date) + 9
        Cal_index = s.find('Calibration') -1
        if voltage != -1 and Cal_index!= -1:  
            voltage_part = s[voltage:Cal_index]
            voltage_value = float(voltage_part.replace('p', '.').replace('v', ''))
            return voltage_value
    elif runtype == 'DecayConstant':
        voltage = s.find(date) + 9
        if voltage != -1 :  
            shifted_voltage = voltage + 4
            if shifted_voltage < len(s):
                voltage_part = s[voltage:shifted_voltage]
                voltage_value = float(voltage_part.replace('p', '.').replace('v', ''))
                return voltage_value
    else:
        voltage = None
        return voltage
            
def check_trigger_mode(runtype):
    if runtype == 'Saturation':
        trigger_mode = 'External'
    elif runtype == 'TimeConstant':
        trigger_mode = 'External'
    elif runtype == 'Calibration':
        trigger_mode = 'External'
    elif runtype == 'LongS2':
        trigger_mode = 'External'
    elif runtype == 'DecayConstant':
        trigger_mode = 'Self'
    elif runtype == 'DarkRate':
        trigger_mode = 'Self'
    else:
        trigger_mode = 'Unknown'
    return trigger_mode


def parse_run_info(rawfilename, runtype):
    # runtype = determine_runtype(rawfilename)
    run_info = []
    date = find_date(rawfilename)
    file_tag = find_file_tag(rawfilename, runtype)
    if runtype != 'DarkRate':
        volate = find_voltage(rawfilename, runtype)
        delta_t =  find_deta_t(rawfilename, runtype)
        trig_rate = find_trig_rate(rawfilename, runtype)
        run_tag = find_run_tag(rawfilename, runtype)       
        run_info = [{
            'date': date,
            'voltage': volate,
            'delta_t': delta_t,
            'trig_rate': trig_rate,
            'run_tag': run_tag,
            'file_tag': file_tag
        }]
    elif runtype == 'DarkRate':
        run_info = [{
            'date': date,
            'file_tag': file_tag
        }]
    return run_info
 
