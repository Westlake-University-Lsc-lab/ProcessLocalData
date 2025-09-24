import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import fit_package 
from matplotlib import cm
cmap = plt.cm.rainbow(np.linspace(0, 1, 9))

import analysis_data
import runinfo
def wf_array(file_list, Channel='Anode', tag=''):
    config = []
    waveform_dictionary={}
    ftag = ''
    with open(file_list, 'r') as list:
        for line in list: 
            file = line.rstrip('\n')
            index = file.find('reference') 
            if tag == 'Rd7':
                if index != -1:
                    ftag = 'Ref'
                    config_tag=file.split('base_')[1].split('_run')[0]                    
                elif index == -1:             
                    ftag = file.split('Rd7_')[1].split('_run')[0]
                    config_tag=file.split('base_')[1].split('_run')[0]                    
            if tag == 'Rd':
                if index != -1:
                    ftag = 'Ref'
                    config_tag=file.split('base_')[1].split('_run')[0]                    
                elif index == -1:             
                    ftag = file.split('base_Rd')[1].split('_Riz')[0]
                    config_tag=file.split('base_')[1].split('_run')[0]
            if tag == 'Riz':
                if index != -1:
                    ftag = 'Ref'
                    config_tag=file.split('base_')[1].split('_run')[0]                    
                elif index == -1:             
                    ftag = file.split('_Riz')[1].split('_Cc10nf')[0]
                    config_tag=file.split('base_')[1].split('_run')[0]
            if tag == 'Dt':
                ftag=file.split('680mv_')[1].split('_50hz')[0]
                config_tag=file.split('combine_')[0]+'combine_'+file.split('combine_')[1].split('680mv_')[0]+'680mv_'
                config_tag = config_tag.split('lv2414_')[1]
            if tag == '':
                print('No tag is given, this is just check wf feature')
                # ftag=file.split('base_')[1].split('_run')[0]
                ftag=file.split('outnpy/')[1].split('.h5py')[0] 
                # config_tag=file.split('outnpy/')[1].split('.h5py')[0]  
                config_tag = runinfo.determine_runtype(file)               
            mean_, std_ = analysis_data.calculate_fullwf_mean_std(file, threshold=100, Channel=Channel)
            waveform_dictionary[ftag] = {'mean_wf':mean_, 'std_wf':std_}
            config.append(config_tag) 
    return waveform_dictionary, config 

def plot_wf_array(flist, Channel='Anode'):
    wf_dic = {}
    with open(flist, 'r') as list:
        for line in list: 
            file = line.rstrip('\n')  
            runtag = runinfo.find_run_tag(file)           
            mean, std = analysis_data.calculate_fullwf_mean_std(file, threshold=100, Channel=Channel)
            wf_dic[runtag] = {'mean':mean,'std':std}            
    i = 0
    for tag, data in wf_dic.items():
        plot_waveform(data['mean'], data['std'], i , tag)
        i +=1
    plt.xlabel('Sample Index[4ns]')
    plt.ylabel('Amplitude[ADC]')
    plt.legend(loc='upper right')   
    plt.show()


def plot_waveform(mean_wf, std_wf, index, label_str):
    """plot waveform 
    parameter:
        mean_wf (np.array): mean value of waveform.
        std_wf (np.array): standard deviation of waveform, same length with mean_wf.
        Channel (str): 'Anode or Dynode'
        LED_config (str): '1p8v_900mv'
    """ 
    x = np.arange(len(mean_wf))  
    plt.fill_between(x, mean_wf - std_wf, mean_wf + std_wf, color=cmap[index], alpha=0.3)  
    plt.plot(x, mean_wf, color=cmap[index], label=label_str)  
    plt.legend(loc='upper right')
    plt.xlabel('Sample Index[4ns]')
    plt.ylabel('Amplitude[ADC]')  
    
def plot_waveform_from_df(df, index, st=0, ed=500, title_str='', save=False):
    """plot waveform from dataframe
    parameter:
        df (pd.DataFrame): dataframe with waveform data.
        index (int): index of the event
        st (int): start index of the waveform
        ed (int): end index of the waveform
        title_str: str, title of the plot
    """
    waveform = df.Wave[index][st:ed]
    baseline = df.Baseline[index]
    x = np.arange(len(waveform))
    plt.step(x, waveform, where='mid', label='data')
    plt.axhline(y=baseline, color='b', linestyle='--', label='Baseline')    
    # plt.axvline(x=df.st[index], color='r', linestyle='--', label='start')    
    # plt.axvline(x=df.ed[index], color='g', linestyle='--', label='end')    
    # plt.axvline(x=df.md[index], color='grey', linestyle='--', label='peak')    
    plt.xlabel('Time [4 ns]')
    plt.title(title_str)
    plt.legend()
    if save:
        plt.savefig(r'figs/{}_example_wf.png'.format(title_str))       
    plt.show()    
    
# def plot_waveform(wave, baseline,  xmin=0, xmax=150, ttt=888, area=100, pmt='LV2414',  ch='Anode', Save=False, file_tag='20240830_LED_run', title='LED Ch0 Waveform'):
#     plt.figure()
#     plt.step(np.arange(len(wave)), wave, where='mid')
#     plt.axhline(y=baseline, color='b', linestyle='--', label='Baseline')  # 基线
#     plt.axvline(x=100, color='r', linestyle='--', label='intg_st')  # 阈值
#     plt.axvline(x=370, color='g', linestyle='--', label='intg_ed')  # 阈值
#     # plt.axvline(x=400, color='r', linestyle='--', label='')  # 阈值
#     # plt.axhline(y=baseline -20, color='r', linestyle='--', label='Threshold')  # 阈值
#     # plt.scatter(st, wave[st], color='r', marker='o', label='start')  # 起始点
#     # plt.scatter(ed, wave[ed], color='g', marker='o', label='end')  # 结束点
#     # plt.scatter(lp, wave[lp], color='b', marker='o', label='minpoint')  # 最低点
#     plt.title(r'Waveform of PMT {} {}, TTT={}, {:.2f}PE'.format(pmt,ch, ttt, area))
#     plt.xlabel('Time [4 ns]')
#     plt.ylabel('ADC Count')
#     plt.legend()
#     plt.xlim(xmin, xmax)
#     if Save==True:
#         plt.savefig(r'./figs/{}_{}.png'.format(file_tag, title,dpi=300))
#     elif Save==False:
#         plt.show()
#     # plt.show()
    
def plot_example_waveform(df,st=0,ed=500):    
    index = None 
    for i in range(3):
        if df.Ch[i] != 0:
            continue
        else:
            index = i
        index = i  
    if 'Wave' not in df.columns:
        raise ValueError("The DataFrame does not contain a 'Wave' column.")
    channel = df.Ch[:][index]
    wave = df.Wave[:][index]
    baseline = df.Baseline[:][index]    
    ttt = df.TTT[:][index]  
    area = df.Area_S2[:][index]   
    if channel == 0:
        pmt = 'LV1414'
    elif channel == 1:
        pmt = 'LV2415'
    elif channel == 2:
        pmt = 'LV2414 Dynode'
    analysis_data.plot_waveform(wave,baseline,st,ed,pmt=pmt,ch=r'Ch={}'.format(channel),ttt=ttt,area=area)

def plot_fit_histgram_vs_Gaussion(array, nbins, left_edge, right_edge, p0=[1.e4, 100, 10],file_tag='20240830_LED_run',xlabel='Ch0 Area', title='LED Ch0 Area', Save=False):
    fig = plt.figure()
    hist, bins_edges = np.histogram(array, bins= nbins, range=(left_edge, right_edge))
    bins = (bins_edges[:-1] + bins_edges[1:])/2
    popt, _ = curve_fit(fit_package.gaussian, bins, hist, p0=p0)
    
    x_fit = np.linspace(np.min(bins), np.max(bins), 1000)
    y_fit = fit_package.gaussian(x_fit, *popt)
    plt.plot(x_fit, y_fit, color='red', label=r'$\mu$={:.2f}, $\sigma$={:.2f}'.format(popt[1], popt[2]))
    plt.hist(array, bins=nbins, range=(left_edge, right_edge),  color='black', density=False, alpha=0.5, label=title)
    plt.xlabel(r'{} '.format(xlabel))
    plt.ylabel('Entries')
    plt.title(r'{} {}'.format(file_tag, title))
    plt.legend()
    if Save==True:
        plt.savefig(r'./figs/{}_{}.png'.format(file_tag, title,dpi=300))
        plt.close(fig)
    elif Save==False:
        plt.show(block=False)
        plt.pause(2)
        plt.close(fig)   
    
    print(r'Fit: mu= {:.2f}, sigma ={:.2f}'.format(popt[1], popt[2]))
    return popt[1], popt[2]

    
def plot_PEns(f, index, DyOption=False ):
    df = pd.read_hdf(f, key='winfo')
    plt.errorbar(df.PEns_filter, df.PEns_anode,
                 xerr=df.PEns_filter_err, yerr= df.PEns_anode_err,
                 fmt='.', markersize=10, color=cmap[index], alpha=0.5, 
                 capsize=3, elinewidth=2, capthick=2, label=df.RunTag.values[0])
    if DyOption is True:
        plt.errorbar(df.PEns_filter, df.PEns_dynode,
                     xerr=df.PEns_filter_err, yerr= df.PEns_dynode_err,
                     fmt='.', markersize=10, color=cmap[index], alpha=0.9, 
                     capsize=3, elinewidth=2, capthick=2, label=df.RunTag.values[0])
        
    plt.ylabel('Saturation-mitigated[PE/ns]')
    plt.xlabel('Monitor[PE/ns]')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(-1.E1, 3.e3)
    plt.ylim(-1.E2, 3.e3)
    plt.legend(loc='upper left')
    plt.show()
    


def plot_2d_histogram(x_series, y_series, 
                     x_range=(0, 500), 
                     y_range=(0, 200),
                     bins=(30, 20),
                     density=True,
                     grid_alpha=0.3,
                     figsize=(10, 8),
                     title=None,
                     xlabel=None,
                     ylabel=None,
                     colorbar_label="频数密度",
                     edgecolor='white',
                     return_data=False,
                     save_path=None,
                     log_z=False,
                     ):
    """
    绘制二维直方图（热力图）

    参数：
    x_series : pandas.Series
        X轴数据序列
    y_series : pandas.Series
        Y轴数据序列
    x_range : tuple, 默认 (0, 500)
        X轴显示范围
    y_range : tuple, 默认 (0, 200)
        Y轴显示范围
    bins : int/tuple, 默认 (30, 20)
        分箱数量（x_bins, y_bins）或统一分箱数
    cmap_colors : list, 可选
        自定义颜色列表（16进制颜色代码）
    density : bool, 默认 True
        是否显示归一化密度（True显示概率密度，False显示频数）
    grid_alpha : float, 默认 0.3
        网格线透明度（0-1）
    figsize : tuple, 默认 (10,8)
        图像尺寸
    title : str, 可选
        标题文本
    xlabel/ylabel : str, 可选
        坐标轴标签
    colorbar_label : str, 默认 "频数密度"
        颜色条标签
    edgecolor : str, 默认 'white'
        分箱边界线颜色
    return_data : bool, 默认 False
        是否返回分箱统计数据
    save_path : str, 可选
        图片保存路径

    返回：
    fig : matplotlib Figure对象
    ax : matplotlib Axes对象
    hist_data : tuple（当return_data=True时）
        (counts, x_edges, y_edges)
    """
    
    from matplotlib.colors import LinearSegmentedColormap
    
    # 数据预处理
    df = pd.DataFrame({'x': x_series, 'y': y_series}).dropna()
    
    # 应用范围过滤（非截断）
    mask = (df['x'].between(*x_range)) & (df['y'].between(*y_range))
    df = df[mask].copy()
    
    # 参数标准化
    if isinstance(bins, int):
        x_bins = y_bins = bins
    else:
        x_bins, y_bins = bins
    
    # 生成直方图数据
    counts, x_edges, y_edges = np.histogram2d(
        df['x'], df['y'],
        bins=[x_bins, y_bins],
        range=[x_range, y_range],
        density=density
    )
    # 对数变换处理
    if log_z:
        # 避免log(0)，加一个极小正数
        counts = np.where(counts > 0, counts, 1e-10)
        counts = np.log10(counts)
        # 修改颜色条标签，提示是对数刻度
        colorbar_label = "log10(" + colorbar_label + ")"
    
    # # 颜色映射
    # default_colors = ["#F0F9E8", "#BAE4BC", "#7BCCC4", "#43A2CA", "#0868AC"]
    # # cmap_colors=["#440154", "#3B528B", "#21918C", "#5DC963"]  # viridis风格
    # cmap = LinearSegmentedColormap.from_list(
    #     "custom_hist", 
    #     cmap_colors if cmap_colors else default_colors
    # )
    
    # 自定义彩虹色节点（示例：增强红色表现）
    custom_rainbow = LinearSegmentedColormap.from_list(
        "strong_red_rainbow",
        ["#4B0082", "#0000FF", "#00FF00", "#FFFF00", "#FF8000", "#FF0000"]
    )
    # if counts < 1000:
    #     cmap  = 'white'
    # elif counts >= 1000:
    cmap = custom_rainbow
    
    # 创建画布
    # fig, ax = plt.subplots(figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')  # 修改1：图形背景
    ax.set_facecolor('white')
    
    # 绘制热力图
    mesh = ax.pcolormesh(x_edges, y_edges, counts.T,
                         cmap=cmap,
                         edgecolor=edgecolor,
                         linewidth=0.5
                         )
    
    # 添加颜色条
    cbar = fig.colorbar(mesh, ax=ax,  pad=0.02)
    cbar.set_label(colorbar_label, rotation=270, labelpad=15)
    
    # 坐标轴设置
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel(xlabel if xlabel else x_series.name)
    ax.set_ylabel(ylabel if ylabel else y_series.name)
    ax.set_title(title if title else f"{x_series.name} vs {y_series.name} 二维分布")
    
    # 网格线
    ax.grid(True, linestyle='--', alpha=grid_alpha)
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return (fig, ax, (counts, x_edges, y_edges)) if return_data else (fig, ax)


'''
# 调用算法
fig, ax = plot_2d_histogram(
        x_series=df['Hight'],
        y_series=df['RFHight'],
        x_range=(0, 1500),
        y_range=(-10, 100),
        bins=(50, 50),
        density=False,
        title="Relection Factor",
        xlabel="Hight[ADC]",
        ylabel="RFhight[ADC]",
        colorbar_label="Counts",
        save_path="figs/reflection_RFHight_nDC_100nF_1p63V_12ns.png"
    )
    
plt.show()
'''


def process_waveforms(df_wave, df_md):
    waveforms = df_wave.tolist()  # 转换为波形列表
    mid_indices = df_md.to_numpy().astype(int)  # 转换为索引数组

    valid_snippets = []
    for wf, mid in zip(waveforms, mid_indices):
        waveform = np.array(wf, dtype=np.float64)  
        length = len(waveform)
        
        # 计算实际截取范围
        start = max(0, mid - 10)
        end = min(length, mid + 20)
        
        # 计算需要填充的长度（确保非负）
        pad_left = max(0, 10 - (mid - start))  # 左边需要填充的点数
        pad_right = max(0, 20 - (end - mid))  # 右边需要填充的点数
        
        # 截取实际波形片段
        snippet = waveform[start:end]
        
        # 执行填充（仅当需要填充时）
        if pad_left > 0 or pad_right > 0:
            snippet = np.pad(snippet, (pad_left, pad_right), 
                            mode='constant', constant_values=np.nan)
        
        valid_snippets.append(snippet)

    stacked_waves = np.array(valid_snippets)
    mean_vals = np.nanmean(stacked_waves, axis=0)
    std_vals = np.nanstd(stacked_waves, axis=0, ddof=1)
    
    return mean_vals, std_vals


def mean_waveforms(df_wave: np.ndarray, start_idx: int, end_idx: int):
    """
    计算多条波形在[start_idx:end_idx]区间内的均值和标准差。
    如果截取区间超出波形边界，用nan填充。

    参数：
        df_wave: np.ndarray, shape (num_waveforms, waveform_length)
        start_idx: int, 截取起始索引（包含）
        end_idx: int, 截取结束索引（不包含）

    返回：
        mean_vals: np.ndarray, shape (end_idx - start_idx,)
        std_vals: np.ndarray, shape (end_idx - start_idx,)
    """
    num_waveforms, waveform_length = df_wave.shape
    snippet_length = end_idx - start_idx
    snippets = np.full((num_waveforms, snippet_length), np.nan, dtype=np.float64)

    for i in range(num_waveforms):
        # 计算有效截取区间
        start = max(start_idx, 0)
        end = min(end_idx, waveform_length)

        # 计算填充区间在snippet中的位置
        snippet_start = start - start_idx  # 可能为0或正数
        snippet_end = snippet_start + (end - start)

        # 将有效波形片段复制到对应位置
        snippets[i, snippet_start:snippet_end] = df_wave[i, start:end]

    mean_vals = np.nanmean(snippets, axis=0)
    std_vals = np.nanstd(snippets, axis=0, ddof=1)

    return mean_vals, std_vals


def analyze_df(df: pd.DataFrame, threshold: float):
    """
    输入：
        df: 包含 'Hight', 'TTT', 'Baseline' 列的 DataFrame
        threshold: Hight 的阈值
    返回：
        delta_pairs: [(delta_ttt_us, delta_baseline), ...] 列表
    并绘制二维分布图，delta_ttt 单位为微秒(us)
    """
    delta_pairs = []

    # 找出所有超过阈值的行索引
    trigger_indices = df.index[df['Hight'] > threshold].tolist()

    if not trigger_indices:
        print("没有超过阈值的 Hight 值。")
        return delta_pairs

    # 添加终止索引，方便遍历最后一段数据
    trigger_indices.append(df.index[-1] + 1)

    for i in range(len(trigger_indices) - 1):
        start_idx = trigger_indices[i]
        end_idx = trigger_indices[i+1]

        # 基准值
        base_ttt = df.at[start_idx, 'TTT']
        base_baseline = df.at[start_idx, 'Baseline']

        # 取当前区间的数据（不包含下一个触发点）
        segment = df.loc[start_idx:end_idx - 1]

        for idx, row in segment.iterrows():
            delta_ttt_ns = (row['TTT'] - base_ttt) * 4  # ns
            delta_ttt_ms = delta_ttt_ns / 1E6           # 转换为 us
            delta_baseline = row['Baseline'] - base_baseline
            delta_pairs.append((delta_ttt_ms, delta_baseline))

    # 绘图
    delta_ttt_vals = [x[0] for x in delta_pairs]
    delta_baseline_vals = [x[1] for x in delta_pairs]

    plt.figure(figsize=(8,6))
    plt.scatter(delta_ttt_vals, delta_baseline_vals, s=5, alpha=0.6)
    plt.xlabel('Delta TTT (ms)')
    plt.ylabel('Delta Baseline(ADC)')
    plt.title('Delta TTT vs Delta Baseline Distribution')
    # plt.savefig('figs/delta_baseline_vs_delta_ttt_1uf.png')
    
    plt.grid(True)
    plt.show()

    return delta_pairs



def plot_baseline_vs_ttt(df: pd.DataFrame):
    """
    绘制 Baseline 随 TTT 变化的图
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['TTT']*4.E-9, df['Baseline'], marker='o', linestyle='-')
    plt.xlabel('TTT(s)')
    plt.ylabel('Baseline(ADC)')
    plt.title('Baseline vs TTT (1uf)')
    plt.ylim(14000,15500)
    plt.xlim(0,100)
    plt.grid(True)
    # plt.savefig('figs/baseline_vs_ttt_1uf.png')
    plt.show()


import pandas as pd
import numpy as np

def calculate_ttt_difference(df):
    """
    计算 DataFrame 中 'TTT' 列的相邻行差值，并返回差值数组
    
    参数:
    df (pd.DataFrame): 输入的 DataFrame，必须包含 'TTT' 列
    
    返回:
    np.array: 包含所有 TTT 差值的一维数组
    
    异常处理:
    - 如果输入不是 DataFrame
    - 如果 DataFrame 中不存在 'TTT' 列
    - 如果 DataFrame 为空或只有一行
    """
    # 检查输入是否为 DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入参数必须是 pandas DataFrame")
    
    # 检查 'TTT' 列是否存在
    if 'TTT' not in df.columns:
        raise ValueError("DataFrame 中不存在列 'TTT'")
    
    # 检查 DataFrame 是否有足够的数据行
    if len(df) < 2:
        print("警告: DataFrame 行数不足，无法计算差值")
        return np.array([])  # 返回空数组
    
    try:
        # 计算相邻行的差值
        delta_ttt = df['TTT'].diff().values
        
        # 移除第一个 NaN 值（因为第一行没有前一行可以计算差值）
        delta_ttt = delta_ttt[1:]
        
        return delta_ttt
        
    except Exception as e:
        print(f"计算差值时发生错误: {e}")
        return np.array([])  # 出错时返回空数组