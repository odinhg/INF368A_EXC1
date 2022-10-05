# INF368A Exercise 1

## How to run things
### Training
- To train the classifier, run *train.py*.
- The best model will be saved as *best.pth* in the *checkpoints* folder. 
- A plot showing both training and validation accuracies and losses will be saved to the root directory as *training_plot.png*.

### Evaluating
- To evaluate the classifier on test data, run *evaluate.py*.
- The total accuracy and accuracy for each class the model is trained on will be printed.

### Embedding
- To compute and save embeddings (activations in the second to last layer) as pickled pandas dataframes, run *embed.py*
- Embeddings are saved in *embeddings_train.pkl*, *embeddings_test.pkl* and *embeddings_unseen.pkl* for train data, test data, and unseen classes, respectively.
- First column is label (index), the rest are activations.

### Configuration
In *configfile.py* one can set the most important settings before training and evaluating the model.

## Architecure / Backbone
The backbone consists of the following:
- 1x (frozen) EfficientNet v2 (using small weights)
- 2x Convolutional layers (with 3x3 kernel) with ReLU activation and batch normalization
- 1x Max pooling layer
- 1x Fully connected layer with ReLU activation
- 1x Drop out layer (p=0.2)
- 1x Fully connected layer

Full specifications can be seen in *backbone.py*.

