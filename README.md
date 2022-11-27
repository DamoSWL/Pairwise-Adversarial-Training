# Pairwise-Adversarial-Training
Pairwise Adversarial Training for Class-imbalanced Domain Adaptation


## Dataset
please download OfficeHome, Domainnet and Office31 and change the `dataset_dir` in the configuration files under config folder 

## Experiment
run experiment on OfficeHome dataset:

    python main.py --config ./config/officehome.yml


run experiment on Domainnet dataset:

    python main.py --config ./config/domainnet.yml


run experiment on Office31 dataset:

    python main.py --config ./config/office31.yml


## Citation
if you think our paper and code is useful, feel free to cite our paper

    @inproceedings{shi2022pairwise,
    title={Pairwise Adversarial Training for Unsupervised Class-imbalanced Domain Adaptation},
    author={Shi, Weili and Zhu, Ronghang and Li, Sheng},
    booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
    pages={1598--1606},
    year={2022}
    }
