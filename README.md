# Blind Concealment from Reconstruction-based Attack Detectors for Industrial Control Systems via Backdoor Attacks
## In Proceedings of the 9th ACM Cyber-Physical System Security Workshop (CPSS ’23)

When using the code from this repository please cite our work as follows:

```
@InProceedings{walita23icsbackdoorattacks,
  title      = {Blind Concealment from Reconstruction-based Attack Detectors for Industrial Control Systems via Backdoor Attacks},
  author     = {Walita, Tim, Erba, Alessandro, John H. Castellanos and Tippenhauer, Nils Ole},
  booktitle  = {Proceedings of the ACM Cyber-Physical System Security Workshop (CPSS)},
  year       = {2023},
  month      = JUL,
  doi        = {10.1145/3592538.3594271},
  publisher  = {ACM},
  address={New York, NY, USA}
}
```

### Requirements
In the following I list the main libraries that I have used in my virtual environment to run all files: <br>
1. tensorflow  (2.2.0)
2. keras  (2.3.1)
3. keras-preprocessing  (1.1.0) 
4. pandas (1.2.2)
5. numpy  (1.19.1)
6. scikit-learn  (0.24.1)
7. scipy  (1.4.1)
8. seaborn  (0.11.1)
9. notebook (6.2.0)
10. jupyter  (1.0.0)
11. h5py  (2.10.0)

### Folder Overview
`Attacks`: Contains all attacks and helper files <br> 
`Evaluation`: Contains the notebook to evaluate the attacks and some graphs <br>
`autoencoder`: Contains the original BATADAL datasets (in `BATADALcsv`) and the class file for the attacked model (autoencoder) <br>
`backdoored_datasets`: Contains all the backdoored datasets that are generated when running an attack

### Standard Backdoor Attack
`python3 Standard_Backdoor_Attack.py` <br>
You can change the pattern/trigger in this attack manually in the main function by changing the list index. I marked the corresponding code line with a comment that starts with "CHANGE ME...".

#### Seeds 
These are the seeds to reconstruct the results mentioned in the thesis for this attack. <br>
Seed in attack file: random seed = 123 <br>
Seed in evaluation file: tensorflow seed = 123


### Improved Standard Backdoor Attack
`python3 Improved_Standard_Attack.py` <br>
You can change the pattern/trigger in this attack manually in the main function by changing the list index. I marked the corresponding code line with a comment that starts with "CHANGE ME...".

#### Seeds
These are the seeds to reconstruct the results mentioned in the thesis for this attack: <br>
Seed in attack file: random seed = 123 <br>
Seed in evaluation file: tensorflow seed = 123


### Combined Backdoor Attack
`python3 File_To_Execute_Combined_Attack.py` <br>
This attack runs automatically on all 51 patterns sequentially and also evaluates each pattern on our own test dataset. In the end, it prints the best result. <br>
The attack can also be run manually. Therefore, the main function needs to be changed accordingly (by commenting and uncommenting the respective lines) in the file `Combined_Backdoor_Attack.py` and this file needs to be executed instead of the previous one.

#### Seeds
These are the seeds to reconstruct the results mentioned in the thesis for this attack: <br>
Seed in attack file: random seed = 123 <br>
Seed in evaluation file: tensorflow seed = 123


### Constrained Backdoor Attack
`python3 Constrained_Backdoor_Attack.py` <br><br>
This attack runs for the second possible pattern of PLC 3 ([1, 0, 0, 0] - best result) on default. You can change the PLC and pattern in the main function manually. At the respective places in the code, I added comments that start with "CHANGE ME..." and I also explain how you can change the PLC or pattern there.

#### Seeds
These are the seeds to reconstruct the results mentioned in the thesis for this attack: <br>
Seed in attack file (PLC 1): random seed = 99 <br>
Seed in attack file (PLC 3): random seed = 123 <br>
Seed in attack file (PLC 5): random seed = 123 <br>
Seed in evaluation file: tensorflow seed = 123



### Sensor-based Backdoor Attack
`python3 Sensor_Backdoor_Attack.py` <br><br>
You can change the pattern/trigger in this attack manually in the main function by changing the list index. I marked the corresponding code line with a comment that starts with "CHANGE ME...".

#### Seeds
These are the seeds to reconstruct the results mentioned in the thesis for this attack: <br>
Seed in attack file: random seed = 123 <br>
Seed in evaluation file: tensorflow seed = 123



### Evaluation
The evaluation for all attacks can be found in the notebook: `Evaluation.ipynb`
