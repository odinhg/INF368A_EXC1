# INF368A Exercise 1

## How to run things
### Training
- To train the classifier, run *train.py*.
- The best model will be saved as *best.pth* in the *checkpoints* folder. 
- A plot showing both training and validation accuracies and losses will be saved to the root directory as *training_plot.png*.

### Evaluating
- To evaluate the classifier on test data, run *evaluate.py*.
- The total accuracy and accuracy for each class the model is trained on will be printed.
- Embeddings (activations in the second to last layer) of the seen classes and unseen classes will be saved to a pandas data frame *embeddings.pkl*

### Configuration
In *configfile.py* one can set the most important settings before training and evaluating the model.

- *class_names*: Names of the classes to train the classifier on.
- *class_names_unseen*: Names of the classes not trained on but to be embedded when evaluating.
- *batch_size*: Batch size
- *epochs*: Epochs to train. Note that early stopping is implemented so this is the maximum number of epochs.
- *lr*: Learning rate (optimizer is Adam).
- *val*: Fraction of data to be used as validation data.
- *test*: Fraction of data to be used as test data.
- *image_size*: Image size going into the model. (Resize is done in the dataloader).
- *device*: CUDA device to use.

## Architecure / Backbone


