# RiskDreamer: Autonomous Driving via Entropy-Risk Balancing Action Expansion in Batch Planning with Trusted Traffic Simulations

<div align="center">
    <a href="https://www.ujs.edu.cn/">Jiangsu University</a> & <a href="https://www.ntu.edu.sg/">Nanyang Technological University</a>
</div>

<!-- <div align="center">

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![PyTorch](https://img.shields.io/badge/torch-2.0-orange)
![Python](https://img.shields.io/badge/python-3.10-blue)
![SUMO](https://img.shields.io/badge/sumo-1.19.0-green)
</div> -->

<div align="center">
Qingchao Liu, Chengzhi Gao, Xiangkun He, Hai Wang, Chen Lv, Yingfeng Cai, Long Chen
</div>

<div align="center">
  <a href="https://github.com/Gaochengzhi/IEEE_template">[paper]</a>
  <a href="https://github.com/Gaochengzhi/sumo_bayesian_calibration">[code1]</a>
  <a href="https://github.com/Gaochengzhi/RiskDreamer">[code2]</a>
  <a href="https://github.com/ADSafetyJointLab/AD4CHE">[data]</a>
  <a href="https://huggingface.co/taitanpascal/riskdreamer_model">[model]</a>
</div>

## Overview
***Abstract***: Ensuring safety and achieving human-level driving
performance remain significant challenges for autonomous vehi-
cles. While model-based reinforcement learning with planners en-
hances sample efficiency and **facilitates** policy exploration, many
such methods rely on planners employing fixed parameters to bal-
ance expected rewards and risks, which limits their adaptability
in dynamic traffic scenarios. Furthermore, studies use simplified
traffic simulations for training and evaluation often results in
algorithms that overfit to homogeneous traffic agents, resulting
in overly optimistic performance. To address these limitations, we
introduce RiskDreamer, a novel framework that employs batch
planning within the latent space of the world model. facilitating
efficiently exploration. Notably, we extend the action space to
incorporate balancing factors as direct action outputs, optimized
at each step. This enables the dynamic weighting of entropy, risk,
and expected reward, achieving adaptable behavior planning in
diverse traffic conditions. Furthermore, RiskDreamer is trained
within a trustworthy traffic scenario generation framework based
on optimization algorithms, capable of producing heterogeneous
traffic agents from real trajectory datasets. The microscopic
behavioral characteristics and macroscopic aggregate metrics of
the generated background agents align with real-world statistical
distributions. Through experimental results, we demonstrate that
our method achieve competitive performance compared to strong
baseline methods. 

## Framework

<div align="center" style="background-color: white;">
  <img src=".assets/frame1.png" alt="Calibration framework of the trustworthy traffic simulation">
</div>

Calibration framework of the trustworthy traffic simulation

<div align="center" style="background-color: white;">
  <img src=".assets/frame2.png" alt="Framework of the RiskDreamer algorithm">
</div>

Framework of the RiskDreamer algorithm. (a) Batch planning within latent space. (b) Action expansion for balancing entropy-risk.

## Results

<div align="center" style="background-color: white;">
  <img src=".assets/train_curve.png" alt="Results of different algorithm in the three scenarios">
</div>

Results of different algorithm in the three scenarios. (a) Training curves comparison. (b) Evaluation results

<div align="center" style="background-color: white;">
  <img src=".assets/scatter_all.png" alt="Evaluation pattern of agent performance">
</div>

Evaluation pattern of agent performance: Speed vs. Navigation. Bigger points indicate denser data point distributions.