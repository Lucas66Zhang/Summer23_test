# Report

## 1. Data Exploration

After observing the input and output data, I find that there are following points need to be noticed:

1. The input data share the same shape as the output data, which means that the number of input rectangular is the same as the number of output rectangular in the same sample.
2. The width and height of the input rectangular and those of the corresponding rectangular are the same (detail shown in `dataset/data_vasualize.ipynb`)

Therefore, I make an inference that the task is to predict the information of positions of several rectangular in the next time step given the current information of positions of them.

## 2. Model design

The structure of the model I designed comes from the above inference.

1. In this case, each input of a sample is the positional information of several rectangular and so does the output of the sample, which is exactly the form of data that the transformer structure can easily handle. Thus, I choose to use multi-head attention layers to construct a simple neural network (that is also why I do not transform the sequence data into image data before making the model learn them).
2. Moreover, as we all know, the multi-head attention layer was first introduced by the transformer to deal with machine translation tasks, which makes it necessary to employ positional encoding to tell the model the order of the words in the sentence. However, in our situation, rectangular in a sample do not have an order. Therefore, positional encoding is not necessary. So I decide to abandon positional encoding in my model.
3. Finally, since the width and height of the rectangular keep unchanged, there is no need to predict them in the next time step. Consequently, we can just drop the two variables in the input and output data.

To sum up, I construct a model with two multi-head attention layers, with embedding_dim equal to 3 (drop weight and height) and without positional encoding. The experiment result shows that just a two-layer model can achieve an ideal performance (The MSE loss on the test is about 0.1317.

## 3. Experimental Results

The model converges quickly on the training set. Moreover, according to the validation result, there is no overfitting problem in this model. The MSE on the test set is about 0.1317.

It should be noticed that there should be another metric that fits the task better, such as the ratio of the area of the predicted rectangle overlapping the target rectangle to the area of the target rectangle. It is more suitable to be used to observe the inference ability of the model more directly. But I think MSE can describe how well the model fit the data as well.

