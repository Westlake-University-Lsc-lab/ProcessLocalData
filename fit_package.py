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
    if channel == 4 :
        area = -area
        mean = np.mean(area)
        std = np.std(area)
        redge = np.max(area)
        ledge = np.min(area)       
        amp = len(area)
    elif channel == 5:
        area = -area
        mean = np.mean(area)
        std = np.std(area)
        redge = np.max(area)
        ledge = np.min(area)       
        amp = len(area)
    else:       
        mean = np.mean(area)
        std = np.std(area)
        ledge = np.min(area)
        redge = np.max(area)    
        amp = len(area)
    nbins = 100        
    if ledge < 0 or ledge < mean/4:
        ledge = mean/4.      
    # if channel == 0:
    #     pmt = 'LV1414'
    # elif channel == 1:
    #     pmt = 'LV2415'
    # elif channel == 2:
    #     pmt = 'LV2414 Dynode'
    tile = r'{}'.format(channel)
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
from scipy.misc import derivative
from scipy.optimize import root_scalar

import numpy as np
from scipy.misc import derivative
from scipy.optimize import root_scalar

def find_extrema(f, x_min, x_max, num_points=1000, dx=1e-6):
    """
    在区间 [x_min, x_max] 内寻找函数 f(x) 的极大值和极小值点。

    参数：
    - f: 一元函数，f(x)
    - x_min, x_max: 搜索区间
    - num_points: 在区间内采样点数，越大精度越高
    - dx: 计算数值导数的步长

    返回：
    - extrema_points: 列表，元素为 (x, f(x), 'max' 或 'min')
    """

    # 在区间内均匀采样
    x_vals = np.linspace(x_min, x_max, num_points)
    # 计算一阶导数
    dy_vals = np.array([derivative(f, xi, dx=dx) for xi in x_vals])

    extrema_points = []

    for i in range(1, len(dy_vals)):
        # 导数符号变化检测
        if dy_vals[i-1] > 0 and dy_vals[i] < 0:
            # 极大值附近，求导数零点
            try:
                sol = root_scalar(lambda x: derivative(f, x, dx=dx), bracket=[x_vals[i-1], x_vals[i]], method='brentq')
                if sol.converged:
                    x_ext = sol.root
                    y_ext = f(x_ext)
                    extrema_points.append((x_ext, y_ext, 'max'))
            except ValueError:
                # 可能区间内无根，跳过
                pass

        elif dy_vals[i-1] < 0 and dy_vals[i] > 0:
            # 极小值附近，求导数零点
            try:
                sol = root_scalar(lambda x: derivative(f, x, dx=dx), bracket=[x_vals[i-1], x_vals[i]], method='brentq')
                if sol.converged:
                    x_ext = sol.root
                    y_ext = f(x_ext)
                    extrema_points.append((x_ext, y_ext, 'min'))
            except ValueError:
                pass

    return extrema_points


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
        df['Area'].values, bins=np.linspace(-10, 60, 100), 
        histtype='step', lw=1,
        label='Area'
    )

    X = bin_value =  (bins[1:] + bins[:-1]) / 2
    Y = count
    
    if case:
        params, params_covariance = curve_fit(three_gauss, X, Y, p0=p0, bounds=bounds)
    else:
        params, params_covariance = curve_fit(four_gauss, X, Y, p0=p0, bounds=bounds)

    param_errors = np.sqrt(np.diag(params_covariance))
    for i, (param, error) in enumerate(zip(params, param_errors)):
        print(f"Param {i}: {param:.4f} ± {error:.4f}")
    # 输出拟合参数
    print("Fitted parameters:")
    print(f"Pedestal: A0={params[0]:.2f}, mu0={params[1]:.2f}± {param_errors[1]:.3f}, sigma0={params[2]:.2f}±{param_errors[2]:.3f}")
    print(f"SPE:      A1={params[3]:.2f}, mu1={params[4]:.2f}±{param_errors[4]:.3f}, sigma1={params[5]:.2f}±{param_errors[4]:.3f}")
    print(f"DPE:      A2={params[6]:.2f}±{param_errors[6]:.3f}")
    if case == False:
        print(f"TPE:      A3={params[7]:.2f}")
    return params,params_covariance, X, Y

# import constant as cts
def plot_three_gauss_fit(df, params, params_covariance,  X, Y, ftag, fstring, save_path=False, case=False):
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
        def f(x):
            return three_gauss(x, *params)
        extrema_points = find_extrema(f, 0, 20, num_points=1000, dx=1e-6)
        print(extrema_points)
        
        max_y_values = [y for x, y, t in extrema_points if t == 'max']
        min_y_values = [y for x, y, t in extrema_points if t == 'min']
        peak_valley_ratio = max_y_values[0]/min_y_values[0]
        print(max_y_values, min_y_values)
        
    elif case == False:
        residuals = Y - four_gauss(X, *params)
        def f(x):
            return four_gauss(x, *params)
        extrema_points = find_extrema(f, 0, 20, num_points=1000, dx=1e-6)
        print(extrema_points)
        max_y_values = [y for x, y, t in extrema_points if t == 'max']
        min_y_values = [y for x, y, t in extrema_points if t == 'min']
        peak_valley_ratio = max_y_values[0]/min_y_values[0]
        print(max_y_values, min_y_values)
        
    perr = np.sqrt(np.diag(params_covariance))
    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Y - np.mean(Y))**2)
    r_squared = 1 - (ss_res / ss_tot)    
    res = sigma1/mu1
    res_err =res* np.sqrt((perr[5]/sigma1)**2  + (perr[1]/mu0)**2)
    print(f"R-squared: {r_squared:.2f}")
        
    textstr = '\n'.join([      
        fr'$\mu_{{Ped}}$:   {mu0:.2f}±{perr[1]:.2f} ',
        fr'$\sigma_{{Ped}}$:   {sigma0:.2f}±{perr[2]:.2f}',
        fr'$\mu_{{SPE}}$:   {mu1:.2f}±{perr[4]:.2f}',
        fr'$\sigma_{{SPE}}$:   {sigma1:.2f}±{perr[5]:.2f}',
        # fr'$R^{2}$:     {r_squared:.2f}',
        fr'Res:    {res:.2f}±{res_err:.2f}',
        fr'Peak-Valley Ratio: {peak_valley_ratio:.2f}'  
])
    # plt.rcParams.update(cts.params)
    plt.figure(figsize=(8, 6))
    plt.hist(df['Area'], bins=100, range=(-10, 60), histtype='step', color='black', alpha=0.8, label='Data')
    if case == False:
        plt.plot(X, four_gauss(X, *params), 'r-', label='')
    elif case == True:
        plt.plot(X, three_gauss(X, *params), 'r-', label='')
    # 绘制各个高斯分量（可选）
    plt.plot(X, A0* np.exp(-(X - mu0)**2 / (2 * sigma0**2)), 'b--', label='')
    plt.plot(X, A1 * np.exp(-(X - mu1 - mu0 )**2 / (2 * (sigma1**2 + sigma0**2)  )), 'g--', label='')
    plt.plot(X, A2 * np.exp(-(X - 2*mu1 - mu0)**2 / (2 *(2*sigma1**2 + sigma0**2)) ), 'm--', label='')
    if case == False:
        plt.plot(X, A3 * np.exp(-(X - 3*mu1 - mu0)**2 / (2 *(3*sigma1**2 + sigma0**2)) ), 'y--', label='')

    plt.ylim(1.E0, 1.E5)

    plt.xlabel("Gain[$10^{6} e$]", labelpad=20, fontsize=20)
    plt.ylabel("Counts", labelpad=20, fontsize=20)
    plt.xticks([0, 10, 20, 30, 40, 50, 60], ['0','10', '20', '30', '40', '50', '60'], fontsize=20)
    
    plt.yticks(fontsize=20)
    plt.tick_params(which='both', direction='in', labelsize=20, pad=7, length=6,width=1.5,)
    
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['right'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)  
    
    plt.text(
        0.6, 0.98, textstr,
        transform=plt.gca().transAxes,       
        fontsize=14,
        ha='left',                    
        va='top',
        bbox=dict(facecolor='none', edgecolor='none')
    )
  
    plt.yscale('log')
    # 保存图像
    if save_path:
        plt.savefig(r'figs/{}_{}_{}_{}.png'.format(ftag, 'SPE', fstring ,df['Ch'].values[0] ),  dpi=300, bbox_inches='tight')
        plt.savefig(r'figs/{}_{}_{}_{}.pdf'.format(ftag, 'SPE', fstring ,df['Ch'].values[0] ),  dpi=300, bbox_inches='tight')
    plt.show()


