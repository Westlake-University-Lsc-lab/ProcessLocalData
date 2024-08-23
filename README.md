# ProcessLocalData
this repository is software to process DAW data from 1725 digitize borad and do some basic calculation for data analysis

#### to install the package, run the following command in terminal:
``` git clone git@github.com:Westlake-University-Lsc-lab/ProcessLocalData.git ```

``` cd ProcessLocalData ```

``` git checkout your_branch_name ```

#### to process data files from mulit runs(LED runs for example), run the following command in terminal:
``` ls -lrth   --time-style=+%Y-%m-%d\ %H:%M  /mnt/data/PMT/R8520_406/LV2415_anodereadout_LV2414_dualreadout_20240821_LED_1.7V_11ns_400_ratio_run* >> runlist/LED_1.7V_11ns_400_ratio_runs ```

####  you need to edit the runlist/LED_1.7V_11ns_400_ratio_runs file with:
``` vim runlist/LED_1.7V_11ns_400_ratio_runs ```

``` ctl+v ```

``` --> right arrow till 'MB', and leave the 'year-month-day-hour-minute '  ```

``` :wq! ```

####  the runlist/LED_1.7V_11ns_400_ratio_runs shows like this:
``` more runlist/LED_1.7V_11ns_400_ratio_runs ```

`` 2024-08-21 14:49 /mnt/data/PMT/R8520_406/LV2415_anodereadout_LV2414_dualreadout_20240821_LED_1.7V_11ns_400_ratio_run0_raw_b0_seg0.bin ``
`` 2024-08-21 15:03 /mnt/data/PMT/R8520_406/LV2415_anodereadout_LV2414_dualreadout_20240821_LED_1.7V_11ns_400_ratio_run1_raw_b0_seg0.bin ``

#### then run the following command to process the data:
``` mkdir outnpy/ ```

``` python bin2h5py.py  runlist/LED_1.7V_11ns_400_ratio_runs  ```
#####  still, there are some bugs need to be fixed, during saving the data to hdf5 format( bin2h5py.py )..
#### the processed data will be saved in hdf5 format in the same directory which you have to run:
``` ls  outnpy/ ```
####  or you can also run ``` raw2h5py.ipynb ```to process the data in jupyter notebook.

#### some data has been processed and saved in hdf5 format, you can found them here:
``` ls /home/yjj/ProcessLocalData/outnpy/ ```

