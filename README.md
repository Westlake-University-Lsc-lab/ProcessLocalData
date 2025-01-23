# ProcessLocalData

this repository is software to process DAW data from 1725 digitize borad and do some basic calculation for data analysis

#### to install the package, run the following command in terminal:

```
git clone [git@github.com](mailto:git@github.com):Westlake-University-Lsc-lab/ProcessLocalData.git
cd ProcessLocalData
git checkout your_branch_name
```

#### to process and analyze the data, you should be under python3.8 environment, the folowing is the steps:

#### convert the binary data to hdf5 format:

```
python bin2h5df.py --runtype ['Saturation or TimeConstant or LongS2 or others'] --file_list {'file_list.txt'}
```

#### use guassian fit the spectrum:

```
python guassainfitspectrum.py.py --file_list {file_list.txt'}
```

#### scale ADC to PEns

```
python sacle2PEns.py --runtype {' Saturation or TimeConstant'} --file {'outnpy/*_single_gussain.h5py'}
```

#### calculate R2ref of the time constant data:

```
python sacle2PEns.py --file  {'outnpy/*single_gussain_TimeConstant_scaled.h5py'}

```
