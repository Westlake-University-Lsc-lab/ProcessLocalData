import pandas as pd
import numpy as np
import analysis_data
import sys
def check_type(input_numb):
    try:
        int_value = int(input_numb)
        print(r'input {} is int type'.format(input_numb))
        return True
    except ValueError:
        if isinstance(input_numb, str):
            print("intput is str")
            return False
        elif isinstance(input_numb, int):
            print(r'input {} is int typ'.format(input_numb))
            return True
        else:
            print("Please input a valid parameter type.")

def ftag(file):
    file_tag = file.split('outnpy/')[1].split('.h5py')[0]
    # fdate = file.split('LED_')[0].split('lv2414_')[1]
    # led_config = file.split('LED_')[1].split('.h5py')[0]
    # file_tag = fdate + led_config
    return file_tag

def read_hdfile(file):
    import glob
    h5_files_pattern = r'{}*.h5py'.format(file.split('raw_')[0])
    print(h5_files_pattern)
    h5_files = glob.glob(h5_files_pattern)
    df = pd.DataFrame()  
    for files in h5_files:
        _df = pd.read_hdf(files, key='winfo')
        df = pd.concat([df, _df], ignore_index=True)
    return df

        
def fit_single_channel(df, channel, ftag):
    if check_type(channel) ==True:
        pass
    else:
        print("Please input a valid parameter type with interger value.")
        sys.exit()   
    area = df.Area[df.Ch == channel].astype(np.float64).to_numpy()
    if channel == 2:
        area = -area
        mean = np.mean(area)
        std = np.std(area)
        redge = np.max(area)
        ledge = np.min(area)       
        amp = len(area)
    elif channel != 2:       
        mean = np.mean(area)
        std = np.std(area)
        ledge = np.min(area)
        redge = np.max(area)    
        amp = len(area)
    nbins = 100        
    if ledge < 0 or ledge < mean/4:
        ledge = mean/4.      
    if channel == 0:
        pmt = 'LV1414'
    elif channel == 1:
        pmt = 'LV2415'
    elif channel == 2:
        pmt = 'LV2414 Dynode'
    tile = r'{}'.format(pmt)
    s2_mu, s2_sigma =  analysis_data.plot_fit_histgram_vs_Gaussion(
        area,nbins,ledge,redge,p0=[amp,mean,std],file_tag=ftag, 
        xlabel=(r'Ch{}Area (PE)'.format(channel)),title=tile,Save=False)
    return s2_mu, s2_sigma
    


def fit_baseline(df, channel, ftag):
    if check_type(channel) ==True:
        pass
    else:
        print("Please input a valid parameter type with interger value.")
        sys.exit()   
    base = df.Baseline[df.Ch == channel].astype(np.float64).to_numpy()
    mean = np.mean(base)
    std = np.std(base)
    redge = np.max(base)
    ledge = np.min(base)
    amp = len(base)
    nbins = 5           
    if channel == 0:
        pmt = 'LV1414'
    elif channel == 1:
        pmt = 'LV2415'
    elif channel == 2:
        pmt = 'LV2414 Dynode'
    tile = r'{}'.format(pmt)
    base_mu, base_sigma =  analysis_data.plot_fit_histgram_vs_Gaussion(
        base,nbins,ledge,redge,p0=[amp,mean,std],file_tag=ftag, 
        xlabel=(r'Ch{}Area (PE)'.format(channel)),title=tile,Save=False)
    return base_mu, base_sigma
   



from scipy.optimize import curve_fit
import numpy as np

def linear_fit(x_data, y_data):

    coefficients = np.polyfit(x_data, y_data, 1) 
    return coefficients
def linear_model(x, m, c):
    return m * x + c

def linear_model_fit(x, y):
    popt, pcov = curve_fit(linear_model, x, y)
    return popt, pcov

def log_model(x, a, b):
    return a * np.log(x) + b
def log_model_fit(x, y):
    popt, pcov = curve_fit(log_model, x, y)
    return popt, pcov

from landaupy import landau   
def landau_fit(xdata, location, scale, A):
    mu = location
    sigma = scale
    return A*landau.pdf(xdata, mu, sigma)

def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))


# import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def three_gauss(x, A0, mu0, sigma0, A1, mu1, sigma1, A2):
    """
    三高斯叠加函数：
    - Pedestal（噪声基底）: 高斯1 (A0, mu0, sigma0)
    - 单光电子峰（SPE）: 高斯2 (A1, mu1, sigma1)
    - 双光电子峰（DPE）: 高斯3 (A2, 2*mu1, sigma2)
    """
    gauss_pedestal = A0 * np.exp(-(x - mu0)**2 / (2 * sigma0**2))
    gauss_spe = A1 * np.exp(-(x - mu1-mu0)**2 / (2 * ( sigma1**2 + sigma0**2 )) ) 
    gauss_dpe = A2 * np.exp(-(x - 2*mu1 - mu0 )**2 / (2 *(2*sigma1**2 + sigma0**2)))
    # gauss_tpe = A3 * np.exp(-(x - 3*mu1 - mu0 )**2 / (2 *(3*sigma1**2 + sigma0**2)))        
    return gauss_pedestal + gauss_spe + gauss_dpe


def four_gauss(x, A0, mu0, sigma0, A1, mu1, sigma1, A2, A3):
    """
    三高斯叠加函数：
    - Pedestal（噪声基底）: 高斯1 (A0, mu0, sigma0)
    - 单光电子峰（SPE）: 高斯2 (A1, mu1, sigma1)
    - 双光电子峰（DPE）: 高斯3 (A2, 2*mu1, sigma2)
    """
    gauss_pedestal = A0 * np.exp(-(x - mu0)**2 / (2 * sigma0**2))
    gauss_spe = A1 * np.exp(-(x - mu1-mu0)**2 / (2 * ( sigma1**2 + sigma0**2 )) ) 
    gauss_dpe = A2 * np.exp(-(x - 2*mu1 - mu0 )**2 / (2 *(2*sigma1**2 + sigma0**2)))
    gauss_tpe = A3 * np.exp(-(x - 3*mu1 - mu0 )**2 / (2 *(3*sigma1**2 + sigma0**2)))        
    return gauss_pedestal + gauss_spe + gauss_dpe + gauss_tpe


# import matplotlib.pyplot as plt
def three_gauss_fit(df, p0, bounds, case=False):
    count, bins, *_ = plt.hist(
        df['Area'].values, bins=np.linspace(-100, 800, 100), 
        histtype='step', lw=1,
        label='Area'
    )

    X = bin_value =  (bins[1:] + bins[:-1]) / 2
    Y = count
    
    if case:
        params, params_covariance = curve_fit(three_gauss, X, Y, p0=p0, bounds=bounds)
    else:
        params, params_covariance = curve_fit(four_gauss, X, Y, p0=p0, bounds=bounds)

    # 输出拟合参数
    print("Fitted parameters:")
    print(f"Pedestal: A0={params[0]:.2f}, mu0={params[1]:.2f}, sigma0={params[2]:.2f}")
    print(f"SPE:      A1={params[3]:.2f}, mu1={params[4]:.2f}, sigma1={params[5]:.2f}")
    print(f"DPE:      A2={params[6]:.2f}")
    if case == False:
        print(f"TPE:      A3={params[7]:.2f}")
    return params, X, Y


def plot_three_gauss_fit(df, params, X, Y, ftag, fstring, save_path=False, case=False):
    A0, mu0, sigma0 = params[0], params[1], params[2]
    A1, mu1, sigma1 = params[3], params[4], params[5]
    A2  = params[6]
    mu2 = 2*mu1 + mu0
    sigma2 = np.sqrt(2*sigma1**2 + sigma0**2)
    if case == False:
        A3 = params[7]  
        mu3 = 3*mu1 + mu0
        sigma3 = np.sqrt(3*sigma1**2 + sigma0**2)
    if case == True:
        residuals = Y - three_gauss(X, *params)
    elif case == False:
        residuals = Y - four_gauss(X, *params)
        
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Y - np.mean(Y))**2)
    r_squared = 1 - (ss_res / ss_tot)    
    res = sigma1/mu1
    
    # perr = bse = np.sqrt(np.diag(params_covariance))
    print(f"R-squared: {r_squared:.2f}")

    plt.figure(figsize=(8, 6))
    plt.hist(df['Area'], bins=100, range=(-100, 800), histtype='step', color='black', alpha=0.8, label='Data')
    if case == False:
        plt.plot(X, four_gauss(X, *params), 'r-', label=f"Fit-$R^{2}$={r_squared:.2f}, res={res:.2f}")
    elif case == True:
        plt.plot(X, three_gauss(X, *params), 'r-', label=f"Fit-$R^{2}$={r_squared:.2f}, res={res:.2f}")
    # 绘制各个高斯分量（可选）
    plt.plot(X, A0* np.exp(-(X - mu0)**2 / (2 * sigma0**2)), 'b--', label=f"$\mu_{{Ped}}$={mu0:.2f},$\sigma_{{Ped}}$={sigma0:.2f}")
    plt.plot(X, A1 * np.exp(-(X - mu1 - mu0 )**2 / (2 * (sigma1**2 + sigma0**2)  )), 'g--', label=f"$\mu_{{SPE}}$={mu1:.2f},$\sigma_{{SPE}}$={sigma1:.2f}")
    plt.plot(X, A2 * np.exp(-(X - 2*mu1 - mu0)**2 / (2 *(2*sigma1**2 + sigma0**2)) ), 'm--', label=f"$\mu_{{DPE}}$={mu2:.2f},$\sigma_{{DPE}}$={sigma2:.2f}")
    if case == False:
        plt.plot(X, A3 * np.exp(-(X - 3*mu1 - mu0)**2 / (2 *(3*sigma1**2 + sigma0**2)) ), 'y--', label=f"$\mu_{{TPE}}$={mu3:.2f},$\sigma_{{TPE}}$={sigma3:.2f}")

    plt.ylim(1.E1, 1.E5)

    plt.xlabel("SPE Gain[ADC*4ns]")

    plt.ylabel("Counts")
    plt.yscale('log')
    plt.title(r'SPE Gain @{}'.format(fstring))
    plt.legend()
    # 保存图像
    if save_path:
        plt.savefig(r'figs/{}_{}_{}_{}.png'.format(ftag, 'SPE', fstring ,df['Ch'].values[0] ),  dpi=300, bbox_inches='tight')
    plt.show()


