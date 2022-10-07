import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import isfile, join
from tqdm import tqdm
from configfile import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

def save_accuracy_plot(accuracies, n_samples, method):
    plt.cla()
    y_min = np.min(accuracies) - 0.1
    y_max = np.max(accuracies) + 0.1
    xi = list(range(len(n_samples)))
    plt.ylim(y_min, y_max)
    plt.plot(xi, accuracies, marker="o", linestyle="--", color="b")
    plt.xlabel("Number of samples trained on")
    plt.ylabel("Test accuracy")
    plt.xticks(xi, n_samples)
    plt.title(method)
    #plt.legend()
    plt.savefig("accuracy_" + method + ".png")

if __name__ == "__main__":
    if not isfile("./embeddings_unseen.pkl"):
        exit("Embeddings for unseen classes not found. Please run embed.py first!")
    
    # Load saved embeddings of the data with unseen classes and split
    df = pd.read_pickle("embeddings_unseen.pkl")
    train, test = train_test_split(df, test_size=0.3, shuffle=True, random_state=0)
    
    svm_classifier = make_pipeline(StandardScaler(), SVC(gamma="auto"))
    linear_classifier = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    k = 5
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    n_samples = []
    svc_accuracies = []
    linear_accuracies = []
    knn_accuracies = []
    for n in tqdm(range(10, len(train), 50)):
        X_train = train.iloc[:n, 2:]
        y_train = train.loc[:, "label_idx"].iloc[:n]
        # Fit models
        svm_classifier.fit(X_train, y_train)
        linear_classifier.fit(X_train, y_train)
        knn_classifier.fit(X_train, y_train)

        # Predict
        X_test = test.iloc[:, 2:]
        y_test = test.loc[:, "label_idx"]
        svc_preds = svm_classifier.predict(X_test)
        linear_preds = linear_classifier.predict(X_test)
        knn_preds = knn_classifier.predict(X_test)

        # Compute accuracies
        svc_accuracy = accuracy_score(y_test, svc_preds)
        linear_accuracy = accuracy_score(y_test, linear_preds)
        knn_accuracy = accuracy_score(y_test, knn_preds)

        svc_accuracies.append(svc_accuracy)
        linear_accuracies.append(linear_accuracy)
        knn_accuracies.append(knn_accuracy)
        n_samples.append(n)

    save_accuracy_plot(svc_accuracies, n_samples, "Support Vector Classifier")
    save_accuracy_plot(linear_accuracies, n_samples, "Linear Classifier")
    save_accuracy_plot(knn_accuracies, n_samples, f"k-Nearest Neighbors (k={k})")

    """ 
    dataset = FlowCamDataSet(class_names_unseen, image_size)

    #Split into train and test data
    val_size = int(0.1 * len(dataset))
    test_size = int(0.3 * len(dataset))
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(420))
    for n in range(100, len(train_dataset), 200):
        print(f"Training on {n} samples.")

    #train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    #test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    """
