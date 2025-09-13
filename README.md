# Reframing Unsupervised Machine Fault Detection as Cross-Modal Alignment via CLIP with Anomaly-Aware Contrastive Loss
This is the official project repository for the papaer "Unsupervised Machinery Fault Detection via CLIP with Anomaly-Aware Contrastive Learning and Sharpness-aware Minimization" by Kangkai Wu , Jingjing Li, Ke Lu (submitted to ICASSP2026).

Abstract: Unsupervised machine fault detection is essential for maintaining production safety and efficiency, yet existing methods remain constrained by limited generalization and reliance on manually defined thresholds. In this study, we reformulate fault detection as a cross-modal alignment problem and propose a CLIP-based framework that aligns vibration-signal images with textual state descriptions. To overcome the lack of fault data, we construct physically interpretable pseudo-faults through amplitude scaling and interval perturbations on raw signals, enabling label-free image–text training. We further introduce an Anomaly-Aware Contrastive Loss (AACL) that enforces inter-class separation while preserving cross-modal alignment, and adopt Sharpness-Aware Minimization (SAM) to mitigate overfitting to pseudo-fault samples and improve robustness. Experiments on three benchmark datasets demonstrate that our method surpasses state-of-the-art approaches.

<img width="1346" height="571" alt="image" src="https://github.com/user-attachments/assets/6c02ed16-ddea-4dd6-a67a-1a85005eeeaa" />



## Usage
You can download the data from this [LINK](https://pan.quark.cn/s/b7806b883a60) , or you can also download it from the official link.


* Create Enviroment

    `conda env create -f environment.yaml -n my_new_env`

* For Training

    `conda activate my_new_env`
  
    `sh run_train.sh`
  
## Results
<img width="1346" height="699" alt="image" src="https://github.com/user-attachments/assets/fbb865c9-e440-41db-8700-abe373ea0775" />

