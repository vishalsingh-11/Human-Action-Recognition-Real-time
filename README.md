# Real Time Human Action Recognition - using TimeSformer

## Overview

This project is an implementation of Real Time Human Action Recognition using pretrained [TimeSformer](https://github.com/facebookresearch/TimeSformer) model. The model is trained on 60 action classes from NTU RGB+D action recognition dataset.

## Dependencies

The dependencies required to run the code are specified in the Project_dependencies.sh file


## Usage
The project can be executed by using the below command.

   ```shell
   Run_action.sh
   ```

## Pre-trained Model

For more information on the pretrained model used in this implementation, please reach out at rbondili@uncc.edu

## Acknowledgments

I would like to express my gratitude to [Dominick Reilly](https://dominickrei.github.io/) for generously providing the pre-trained TimeSformer model. The TimeSformer model plays a crucial role in the implementation and contributes to the success of this project. We appreciate the efforts and contributions made by Dominick Reilly in developing and sharing this valuable resource. Please visit their website for more information and explore their other works.

The architecture utilized for this project is based on the TimeSformer model developed by Facebook. The TimeSformer model has been adapted and incorporated into this implementation.

```@inproceedings{gberta_2021_ICML,
    author  = {Gedas Bertasius and Heng Wang and Lorenzo Torresani},
    title = {Is Space-Time Attention All You Need for Video Understanding?},
    booktitle   = {Proceedings of the International Conference on Machine Learning (ICML)}, 
    month = {July},
    year = {2021}
}

```@misc{fan2020pyslowfast,
  author =       {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title =        {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year =         {2020}
}

```@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
