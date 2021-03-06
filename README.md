- [language_model_from_scratch_pytorch](#language_model_from_scratch_pytorch)
- [Brief introduction](#brief-introduction)
- [For developers](#for-developers)
- [LOGS](#logs)
  - [2022/2/26](#2022226)
  - [2022/2/27](#2022227)
  - [2022/2/28](#2022228)
    - [addition](#addition)
  - [2022/3/1](#202231)
    - [addition](#addition-1)
- [Credits](#credits)
# language_model_from_scratch_pytorch
a manageable and trainable language model in PyTorch

This is my first attempt to build a language model in PyTorch in preparation for the NLP class of the course ***Introduction to AI*** in Peking University.

- - -
# Brief introduction
The project contains 2 Python scripts:
- [dataload.py](dataload.py): A dataset class derived from `torch.utils.data.Dataset`, you can choose your txt file and load it. By the way, it can also generate word vectors (although we might not use them).
- [lgg_model.py](lgg_model.py): A language model class, which now contains a vanilla LSTM model, you can add your own model classes to it and use them in the [train.ipynb](train.ipynb) below.

also, there are 2 jupyter notebooks:
- [train.ipynb](train.ipynb): A notebook for training. You can choose different data and use [dataload.py](dataload.py) to load it and choose different models to train. Be sure to have installed tensorboard to see the training process.
- [generate.ipynb](generate.ipynb): A notebook for generating text. To run this notebook, you need a word model trained and saved in folder `word_model_paths` and a language model trained and saved in folder `lgg_model_paths`.

- - - 
# For developers
Feel free to change the hyperparameters and optimize the language model.

The main problem of this model is that it ~~lacks learning rate decay~~ and batch normalization, which I will add in the near future.

- - -
# LOGS
## 2022/2/26
First developed this model, which needs more refinement.
## 2022/2/27
Add a **learning rate decay scheduler** to the optimizer, which leads to better results.

Also, for larger corpus, we can increase the hidden size from 50 to 100 to increase the number of parameters, which will enhance the expressiveness of the model and decrease the loss.

In the near future, more features, such as new model architectures, jieba based word separation and more powerful optimizing techniques, will be added.
## 2022/2/28
Add dropout and batchnorm, but batchnorm isn't working, so I delete it temporarily.

Also, I divide the dataset to trainset and validationset, and use the validationset to see whether the model is overfitting.

In summary, the overall effect of the change to the model architecture is not so satisfying.

### addition
Add a vanilla_GRU model, which is simpler, but less overfitting.
## 2022/3/1
**Modify** the code for **validation**, train a few models.

Merge datasets and add a new dataset.

Try out multiple lr decay methods and LSTM layers and watch their effects.

Learn how to use tensorboard effectively.
### addition
At first, I thought the lr of 0.003 is too big, so I change it into 0.0015, 0.001, 1e-4 and even 1e-5. But it was a mistake. It performs worse and worse. 

To my surprise, when I change lr back to 0.005 (a little more than 0.003), it performs better.

The reason is unclear, but I think it results from the complexity of the model.

- - -

Anyway, from 2022/3/2, the project may not be updated anymore because I am preparing to launch a few new projects. Maybe at some point of the class, I will take it out, make some modifications and update it.

***Stay tuned.***

- - -
# Credits
Thanks for Zhihu user **[@?????????](https://www.zhihu.com/people/bing-feng-ruo-qian)** for the inspiration of this LSTM model.

To learn more, check out his repository: [clickhere](https://github.com/hhiim/Lacan)