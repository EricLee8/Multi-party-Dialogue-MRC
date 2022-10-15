# Codes for EMNLP2021
Codes for our paper Self- and Pseudo-self-supervised Prediction of Speaker and Key-utterance for Multi-party Dialogue Reading Comprehension
## Environments
Our experiments are conducted on the following evironmental settings. To ensure reproductivity, we strongly recommand that you run our code on the same settings.
- GPU: TITAN RTX 24G
- CUDA Version: 10.2
- Pytorch Version: 1.6.0
- Python Version: 3.8.5
## Usage
To install the dependencies, run:

`$ pip install -r requirements.txt`
### FriendsQA

To run experiments on FriendsQA dataset with the default best hyper-parameter settings, run:

`$ cd friendsqa`

`$ unzip data.zip`

`$ python3 myTrain.py --model_num [0|1]`

Here model_num 0 is the baseline model and model_num 1 is our model. 

Due to some stochastic factors(e.g., GPU and environment), it may need some slight tuning of the hyper-parameters using grid search to reproduce the results reported in our paper. Here are the suggested hyper-parameter settings:

- mha_layer_nums: [3, 4, 5]
- learning_rate: [2e-6, 4e-6, 6e-6, 8e-6]

### Molweni
To run experiments on Molweni dataset with the default best hyper-parameter settings, run:

`$ cd molweni`

`$ unzip data.zip`

`$ python3 myTrain.py --model_num [0|1]`

Arguments here is the same as above.

Suggested hyper-parameter settings for grid search:

- mha_layer_nums: [3, 4, 5]
- learning_rate: [8e-6, 1e-5, 1.2e-5, 1.4e-5]

## Citation
If you find our paper and this repository useful, please cite us in your paper:
```
@inproceedings{li-zhao-2021-self-pseudo,
    title = "Self- and Pseudo-self-supervised Prediction of Speaker and Key-utterance for Multi-party Dialogue Reading Comprehension",
    author = "Li, Yiyang  and
      Zhao, Hai",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.176",
    doi = "10.18653/v1/2021.findings-emnlp.176",
    pages = "2053--2063",
}
```
