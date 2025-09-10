# Reframing Unsupervised Machine Fault Detection as Cross-Modal Alignment via CLIP with Anomaly-Aware Contrastive Loss
This is the official project repository for the papaer "Unsupervised Machinery Fault Detection via CLIP with Anomaly-Aware Contrastive Learning and Sharpness-aware Minimization" by Kangkai Wu , Jingjing Li, Ke Lu (submitted to ICASSP2026).

Abstract:Unsupervised machine fault detection is hindered by scarce fault data and the mismatch between synthetic and real anomalies. We reformulate fault detection as a cross-modal alignment problem and propose a CLIP-based framework that aligns vibration-signal images with textual state descriptions. To overcome the lack of fault data, we construct physically interpretable pseudo-faults through amplitude scaling and interval perturbations on raw signals, enabling label-free image–text training. We further introduce an Anomaly-Aware Contrastive Loss (AACL) that enforces inter-class separation while preserving cross-modal alignment, and adopt Sharpness-Aware Minimization (SAM) to mitigate overfitting to pseudo-fault artifacts and improve robustness. Experiments on three benchmark datasets demonstrate that our method surpasses state-of-the-art approaches.

<img width="1343" height="566" alt="image" src="https://github.com/user-attachments/assets/30c6e855-f412-49e5-ac62-e1333c7e9e97" />


## Usage
You can download the data from this [LINK](https://pan.quark.cn/s/b7806b883a60) , or you can also download it from the official link.


* Conda Enviroment

    `conda env create -f environment.yaml -n my_new_env`

* For Training

    `conda activate my_new_env`
    `sh run_train.sh`
  
## Results
<img width="837" height="438" alt="image" src="https://github.com/user-attachments/assets/75c9d29f-981a-480f-9f39-a83bcbc2f5b9" />
