# ProcessLocalData

this repository is software to process DAW data from 1725 digitize borad and do some basic calculation for data analysis

#### to install the package, run the following command in terminal:

```
git clone [git@github.com](mailto:git@github.com):Westlake-University-Lsc-lab/ProcessLocalData.git
cd ProcessLocalData
git checkout your_branch_name
```

#### to process and analyze the data, you should be under **python3.8** environment, the folowing is the steps:

#### convert the binary data to hdf5 format:

```
python bin2h5df.py --runtype ['Saturation/TimeConstant/LongS2/others'] --file_list {'runlist'}
python bin2h5df.py   --runtype Saturation  --file_list runlist/resistor_62p5M_saturation
```

### process file list will be saved in '_runlist/resistor_62p5M_saturation_processed'_

#### use guassian fit the spectrum:

```
python guassainfitspectrum.py.py --file_list {processed_list'}
python guassainfitspectrum.py  --file_list runlist/resistor_62p5M_saturation_processed
```

#### the all fitted data will be saved in 'outnpy/\*\_single_gussain.h5py'

#### scale ADC to PEns

```
python sacle2PEns.py --runtype {' Saturation or TimeConstant'} --file {'outnpy/*_single_gussain.h5py'}
```

#### calculate R2ref of the time constant data:

```
python CalR2ref.py --file  {'outnpy/*single_gussain_TimeConstant_scaled.h5py'}

```

#### calculate the waveform duration of  50% reduce from the height,
#### using mean waveform and low-pass filter algorithm

```
python meanwf.py  --wftype anode --file_list  runlist/r_anode_processed

```