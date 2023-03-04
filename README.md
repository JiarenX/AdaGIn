# Domain Adaptive Graph Infomax via Conditional Adversarial Networks

This is our implementation for the following paper:

>[Jiaren Xiao, Quanyu Dai, Xiaochen Xie, Qi Dou, Ka-Wai Kwok, and James Lam. "Domain Adaptive Graph Infomax via Conditional Adversarial Networks." IEEE Transactions on Network Science and Engineering 10, no. 1 (2023): 35-52.](https://ieeexplore.ieee.org/abstract/document/9866817)


## Abstract
The emerging graph neural networks (GNNs) have demonstrated impressive performance on the node classification problem in complex networks. However, existing GNNs are mainly devised to classify nodes in a (partially) labeled graph. To classify nodes in a newly-collected unlabeled graph, it is desirable to transfer label information from an existing labeled graph. To address this cross-graph node classification problem, we propose a graph infomax method that is domain adaptive. Node representations are computed through neighborhood aggregation. Mutual information is maximized between node representations and global summaries, encouraging node representations to encode the global structural information. Conditional adversarial networks are employed to reduce the domain discrepancy by aligning the multimodal distributions of node representations. Experimental results in real-world datasets validate the performance of our method in comparison with the state-of-the-art baselines.

## Environment requirement
The codes can be run with the below packages:
* python == 3.7.9
* torch == 1.7.1+cu101
* numpy == 1.15.4
* networkx == 1.9.1
* scipy == 1.5.4

## Examples to run the codes
* Transfer task D -> A
```
python AdaGIn.py --epochs 50 --lr_cly 0.010 --aggregator_class mean --output_dims 512,256,64 --arch_disc 512-64-16 --dgi_param 0.1 --source_dataset dblpv7 --target_dataset acmv9 --n_samples 10,10,10
```

## Citation 
If you would like to use our code, please cite:
```
@article{xiao_domain_2023,
  author={Xiao, Jiaren and Dai, Quanyu and Xie, Xiaochen and Dou, Qi and Kwok, Ka-Wai and Lam, James},
  journal={IEEE Transactions on Network Science and Engineering}, 
  title={Domain Adaptive Graph Infomax via Conditional Adversarial Networks}, 
  year={2023},
  volume={10},
  number={1},
  pages={35-52},
  doi={10.1109/TNSE.2022.3201529}}
```
