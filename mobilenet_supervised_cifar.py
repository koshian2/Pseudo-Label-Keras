from keras.layers import Input, UpSampling2D, Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.models import Model

from keras.utils import to_categorical
from keras.datasets import cifar10

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pickle, os, zipfile, glob

#1から訓練させる
def create_cnn():
    net = MobileNet(input_shape=(128,128,3), weights=None, include_top=False)
    # upsampling(32->128)
    input = Input((32,32,3))
    x = UpSampling2D(4)(input)
    x = net(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation="softmax")(x)

    model = Model(input, x)
    model.summary()
    return model

def train(n_labeled_data):
    model = create_cnn()
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)

    y_test_true = np.ravel(y_test)
    X_train = X_train[indices[:n_labeled_data]] / 255.0
    X_test = X_test / 255.0
    y_train = to_categorical(y_train[indices[:n_labeled_data]], 10)
    y_test = to_categorical(y_test, 10)
    
    model.compile("adam", loss="categorical_crossentropy", metrics=["acc"])

    if not os.path.exists("result_mobilenet"):
        os.mkdir("result_mobilenet")

    hist = model.fit(X_train, y_train, batch_size=min(n_labeled_data, 256), 
                     validation_data=(X_test, y_test), epochs=100).history

    with open(f"result_mobilenet/history_{n_labeled_data:05}.dat", "wb") as fp:
        pickle.dump(hist, fp)

    # tsne-plot
    emb_model = Model(model.input, model.layers[-2].output)
    embedding = emb_model.predict(X_test)
    proj = TSNE(n_components=2).fit_transform(embedding)
    cmp = plt.get_cmap("tab10")
    plt.figure()
    for i in range(10):
        select_flag = y_test_true == i
        plt_latent = proj[select_flag, :]
        plt.scatter(plt_latent[:,0], plt_latent[:,1], color=cmp(i), marker=".")
    plt.savefig(f"result_mobilenet/embedding_{n_labeled_data:05}.png")


if __name__ == "__main__":
    n_batches = [500, 1000, 5000, 10000]
    for nb in n_batches:
        train(nb)

    with zipfile.ZipFile("result_mobilenet.zip", "w") as zip:
        for f in glob.glob("result_mobilenet/*"):
            zip.write(f)
