from keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, AveragePooling2D, Dropout
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback

from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np
import pickle, os, zipfile, glob

def basic_conv_block(input, chs, rep):
    x = input
    for i in range(rep):
        x = Conv2D(chs, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x

def create_cnn():
    input = Input(shape=(32,32,3))
    x = basic_conv_block(input, 64, 3)
    x = AveragePooling2D(2)(x)
    x = basic_conv_block(x, 128, 3)
    x = AveragePooling2D(2)(x)
    x = basic_conv_block(x, 256, 3)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation="softmax")(x)

    model = Model(input, x)
    return model

class PseudoCallback(Callback):
    def __init__(self, model, n_labeled_sample, batch_size):
        self.n_labeled_sample = n_labeled_sample
        self.batch_size = batch_size
        self.model = model
        self.n_classes = 10
        # labeled_unlabeledの作成
        (X_train, y_train), (self.X_test, self.y_test) = cifar10.load_data()
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        self.X_train_labeled = X_train[indices[:n_labeled_sample]]
        self.y_train_labeled = y_train[indices[:n_labeled_sample]]
        self.X_train_unlabeled = X_train[indices[n_labeled_sample:]]
        self.y_train_unlabeled_groundtruth = y_train[indices[n_labeled_sample:]]
        # unlabeledの予測値
        self.y_train_unlabeled_prediction = np.random.randint(
            10, size=(self.y_train_unlabeled_groundtruth.shape[0], 1))
        # steps_per_epoch
        self.train_steps_per_epoch = X_train.shape[0] // batch_size
        self.test_stepes_per_epoch = self.X_test.shape[0] // batch_size
        # unlabeledの重み
        self.alpha_t = 0.0
        # labeled/unlabeledの一致率推移
        self.unlabeled_accuracy = []
        self.labeled_accuracy = []

    def train_mixture(self):
        # 返り値：X, y, フラグ
        X_train_join = np.r_[self.X_train_labeled, self.X_train_unlabeled]
        y_train_join = np.r_[self.y_train_labeled, self.y_train_unlabeled_prediction]
        flag_join = np.r_[np.repeat(0.0, self.X_train_labeled.shape[0]),
                         np.repeat(1.0, self.X_train_unlabeled.shape[0])].reshape(-1,1)
        indices = np.arange(flag_join.shape[0])
        np.random.shuffle(indices)
        return X_train_join[indices], y_train_join[indices], flag_join[indices]

    def train_generator(self):
        while True:
            X, y, flag = self.train_mixture()
            n_batch = X.shape[0] // self.batch_size
            for i in range(n_batch):
                X_batch = (X[i*self.batch_size:(i+1)*self.batch_size]/255.0).astype(np.float32)
                y_batch = to_categorical(y[i*self.batch_size:(i+1)*self.batch_size], self.n_classes)
                y_batch = np.c_[y_batch, flag[i*self.batch_size:(i+1)*self.batch_size]]
                yield X_batch, y_batch

    def test_generator(self):
        while True:
            indices = np.arange(self.y_test.shape[0])
            np.random.shuffle(indices)
            for i in range(len(indices)//self.batch_size):
                current_indices = indices[i*self.batch_size:(i+1)*self.batch_size]
                X_batch = (self.X_test[current_indices] / 255.0).astype(np.float32)
                y_batch = to_categorical(self.y_test[current_indices], self.n_classes)
                y_batch = np.c_[y_batch, np.repeat(0.0, y_batch.shape[0])] # flagは0とする
                yield X_batch, y_batch

    def loss_function(self, y_true, y_pred):
        y_true_item = y_true[:, :self.n_classes]
        unlabeled_flag = y_true[:, self.n_classes]
        entropies = categorical_crossentropy(y_true_item, y_pred)
        coefs = 1.0-unlabeled_flag + self.alpha_t * unlabeled_flag # 1 if labeled, else alpha_t
        return coefs * entropies

    def accuracy(self, y_true, y_pred):
        y_true_item = y_true[:, :self.n_classes]
        return categorical_accuracy(y_true_item, y_pred)

    def on_epoch_end(self, epoch, logs):
        # alpha(t)の更新
        if epoch < 10:# 20-80にしたのは？
            self.alpha_t = 0.0
        elif epoch >= 70:
            self.alpha_t = 3.0
        else:
            self.alpha_t = (epoch - 10.0) / (70.0-10.0) * 3.0
        # unlabeled のラベルの更新
        self.y_train_unlabeled_prediction = np.argmax(
            self.model.predict(self.X_train_unlabeled), axis=-1,).reshape(-1, 1)
        y_train_labeled_prediction = np.argmax(
            self.model.predict(self.X_train_labeled), axis=-1).reshape(-1, 1)
        # ground-truthとの一致率
        self.unlabeled_accuracy.append(np.mean(
            self.y_train_unlabeled_groundtruth == self.y_train_unlabeled_prediction))
        self.labeled_accuracy.append(np.mean(
            self.y_train_labeled == y_train_labeled_prediction))
        print("labeled / unlabeled accuracy : ", self.labeled_accuracy[-1],
            "/", self.unlabeled_accuracy[-1])

    def on_train_end(self, logs):
        y_true = np.ravel(self.y_test)
        emb_model = Model(self.model.input, self.model.layers[-2].output)
        embedding = emb_model.predict(self.X_test / 255.0)
        proj = TSNE(n_components=2).fit_transform(embedding)
        cmp = plt.get_cmap("tab10")
        plt.figure()
        for i in range(10):
            select_flag = y_true == i
            plt_latent = proj[select_flag, :]
            plt.scatter(plt_latent[:,0], plt_latent[:,1], color=cmp(i), marker=".")
        plt.savefig(f"result_pseudo/embedding_{self.n_labeled_sample:05}.png")


def train(n_labeled_data):
    model = create_cnn()
    
    pseudo = PseudoCallback(model, n_labeled_data, min(512, n_labeled_data))
    model.compile("adam", loss=pseudo.loss_function, metrics=[pseudo.accuracy])

    if not os.path.exists("result_pseudo"):
        os.mkdir("result_pseudo")

    hist = model.fit_generator(pseudo.train_generator(), steps_per_epoch=pseudo.train_steps_per_epoch,
                               validation_data=pseudo.test_generator(), callbacks=[pseudo],
                               validation_steps=pseudo.test_stepes_per_epoch, epochs=1).history
    hist["labeled_accuracy"] = pseudo.labeled_accuracy
    hist["unlabeled_accuracy"] = pseudo.unlabeled_accuracy

    with open(f"result_pseudo/history_{n_labeled_data:05}.dat", "wb") as fp:
        pickle.dump(hist, fp)

if __name__ == "__main__":
    n_batches = [500, 1000, 5000, 10000]
    for nb in n_batches:
        print(nb, "Starts")
        train(nb)

    with zipfile.ZipFile("result_pseudo.zip", "w") as zip:
        for f in glob.glob("result_pseudo/*"):
            zip.write(f)
