## Description

This repository is aimed at bridging the gap of leveraging object centric learning **from** *generic object classes in synthetic datasets* **to** *application specific, targeted detection (road users) in autonomous driving datasets*. Specifically, we primarily focus on enhancing the implementation transparency of leveraging these models to various real-world autonomous driving data based on the SAVi++ model which reports promising results for such real-world scaling of slot-based object centric learning for video (https://github.com/google-research/slot-attention-video/) [1]


## Pre-requisites for execution

- Create the environment for installing the required libraries and dependencies using the provided **requirements.yaml** file. 

- Depth-Anything-V2 model must be cloned (https://github.com/DepthAnything/Depth-Anything-V2) [2]


## Repository structure

We divide the repository into three workspaces, each of which provides implementations on creating a format-specific custom TFDS and corresponding SAVi++ architecture (including the dataloader) for training a model for emergent object segmentation of road users (vehicles, pedestrians, cyclists):
    - **waymo_ws**: Waymo Open Perception dataset [3]
    - **brno-we_ws**: BRNO Urban Dataset: Winter Extension [4]
    - **ADD_ws**: empty workspace with guidelines for adapting to a custom autonomous driving dataset of user preference 

Below is the complete organization of the current repository (including the model for monocular depth estimation):


## Execution

Refer to the workspace-specific README files


## Other notes

- TensorBoard can be used to visualize the progress of training for both datasets (e.g. `tensorboard --logdir=ckpt_waymo/`)  


## References

[1] @article{elsayed2022savi++,
  title={Savi++: Towards end-to-end object-centric learning from real-world videos},
  author={Elsayed, Gamaleldin and Mahendran, Aravindh and Van Steenkiste, Sjoerd and Greff, Klaus and Mozer, Michael C and Kipf, Thomas},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={28940--28954},
  year={2022}
}

[2] @inproceedings{yang2024depth,
  title={Depth anything: Unleashing the power of large-scale unlabeled data},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10371--10381},
  year={2024}
}

[3] @inproceedings{sun2020scalability,
  title={Scalability in perception for autonomous driving: Waymo open dataset},
  author={Sun, Pei and Kretzschmar, Henrik and Dotiwalla, Xerxes and Chouard, Aurelien and Patnaik, Vijaysai and Tsui, Paul and Guo, James and Zhou, Yin and Chai, Yuning and Caine, Benjamin and others},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={2446--2454},
  year={2020}
}

[4] @inproceedings{ligocki2020brno,
  title={Brno urban dataset-the new data for self-driving agents and mapping tasks},
  author={Ligocki, Adam and Jelinek, Ales and Zalud, Ludek},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={3284--3290},
  year={2020},
  organization={IEEE}
}
