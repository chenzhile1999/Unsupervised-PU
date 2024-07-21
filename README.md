# Unsupervised Deep Unrolling Networks for Phase Unwrapping
Here we provide the official implementation of the CVPR-24 paper, unsupervised deep unrolling networks for phase unwrapping.

## Information
- Authors: Zhile Chen (cs_zhilechen@mail.scut.edu.cn); Yuhui Quan (csyhquan@scut.edu.cn); Hui Ji (matjh@nus.edu.sg)
- Institutes: School of Computer Science and Engineering, South China University of Technology; Department of Mathematics, National University of Singapore
- For any question, please send to **cs_zhilechen@mail.scut.edu.cn**
- For more information, please refer to: [[website]](https://csyhquan.github.io/)

## Requirements
Here lists the essential packages needed to run the script:
* python 3.7.15
* pytorch 1.8.1
* numpy 1.21.6

## Start Training
1. Download the training and test datasets provided in the [Google Drive](https://drive.google.com/drive/folders/1n-4SurREPYGH2s-Dn-OBgvXrcptuaXxG?usp=sharing). Place them under the directory './data', e.g., './data/MoGR training data.hdf5'.
2. Run the training script, e.g.,
```
python train.py --lr 1e-3 --batch_size 10 --stage_num 3 --start_epoch 0 --distil_epoch 200 --end_epoch 700 --scheduler 'exp' --gamma 0.99 --expe_name 'PU_MoGR_Train' --traindata_id 'MoGR training data'
```

## Test
Directly run the test script, e.g.,
```
python test.py --batch_size 10 --stage_num 3 --model_id 'PU_MoGR_Train/params_dict_epoch700.pth' --testdata_id 'MoGR test data_10dB' --save
```

## Citation
```
@inproceedings{chen2024unsupervised,
  title={Unsupervised Deep Unrolling Networks for Phase Unwrapping},
  author={Chen, Zhile and Quan, Yuhui and Ji, Hui},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={25182--25192},
  year={2024}
}
```