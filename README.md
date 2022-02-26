# language_model_from_scratch_pytorch
a manageable and trainable language model in PyTorch

This is my first attempt to build a language model in PyTorch in preparation for the NLP class of the course *Introduction to AI* in Peking University.

- - -
The project contains 2 Python scripts:
- `dataload.py`: A dataset class derived from `torch.utils.data.Dataset`, you can choose your txt file and load it. By the way, it can also generate word vectors (although we might not use them).
- `lgg_model.py`: A language model class, which now contains a vallina LSTM model, you can add your own model classes to it and use them in the `train.ipynb` below.

also, there are 2 jupyter notebooks:
- `train.ipynb`: A notebook for training. You can choose different data and use `dataload.py` to load it and choose different models to train. Be sure to have installed tensorboard to see the training process.
- `generate.py`: A notebook for generating text. To run this notebook, you need a word model trained and saved in folder `word_model_paths` and a language model trained and saved in folder `lgg_model_paths`.

- - - 
Feel free to change the hyperparameters and optimize the language model.

The main problem of this model is that it lacks learning rate decay and batch normalization, which I will add in the near future.