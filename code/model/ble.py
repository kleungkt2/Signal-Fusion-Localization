from util.general_util import get_closest_grid

import tensorflow as tf
import pickle
import os
import numpy as np
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import mutual_info_classif


class AutoEncoder(tf.Module):
    def __init__(self, input_dim, embedding_dim=64):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.encoder = self.__encoder()
        self.decoder = self.__decoder()

    def __encoder(self):
        model = tf.keras.Sequential([
                    tf.keras.layers.Dense(self.embedding_dim*4, input_shape=(self.input_dim,), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=1e-6)),
                    tf.keras.layers.Dense(self.embedding_dim*2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=1e-5)),
                    tf.keras.layers.Dense(self.embedding_dim, kernel_regularizer=tf.keras.regularizers.l2(l=1e-5)),
                ])
        return model

    def __decoder(self):
        model = tf.keras.Sequential([
                    tf.keras.layers.Dense(self.embedding_dim*2, input_shape=(self.embedding_dim,), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=1e-6)),
                    tf.keras.layers.Dense(self.embedding_dim*4, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=1e-5)),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(self.input_dim, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l=1e-5)),
                ])
        return model


class BLENetwork(tf.Module):
    # Create model
    def __init__(self, input_dim, rp_locs, rp_features, n_layers=1, embedding_dim=32, mask_fraction=0.1,
                 noise_stddev=0.15, noise_ratio=0.7, alpha=1, beta=1e-4):
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.noise_stddev = noise_stddev
        self.mask_fraction = mask_fraction
        self.noise_ratio = noise_ratio
        self.rp_locs = rp_locs
        self.rp_features = rp_features
        self.alpha = alpha
        self.beta = beta
        self.layers = [AutoEncoder(input_dim, embedding_dim) for _ in range(n_layers)]
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizers = [tf.keras.optimizers.Adam(0.005) for _ in range(n_layers + 1)]  # 0,...,n-1 for layers, n for fine-tune

    def pretrain_wrapper(self):
        @tf.function
        def pretrain(x, idx):
            if tf.random.uniform([]) > self.noise_ratio:
                xx = tf.where(tf.random.uniform(x.shape) < self.mask_fraction, 0.0, x)
            else:
                xx = tf.clip_by_value(x + tf.random.normal(x.shape, mean=0, stddev=self.noise_stddev), 0, 1)

            for i in range(idx):
                ae = self.layers[i]
                xx = ae.decoder(ae.encoder(xx, training=False), training=False)

            with tf.GradientTape() as tape:
                ae = self.layers[idx]
                xx = ae.decoder(ae.encoder(xx), training=True)

                r_loss = (tf.add_n(ae.encoder.losses) + tf.add_n(ae.decoder.losses))
                r_loss *= self.alpha
                t_loss = self.loss_object(x, xx)
                loss = t_loss + r_loss

                trainable_variables = ae.trainable_variables
                gradients = tape.gradient(loss, trainable_variables)
                self.optimizers[idx].apply_gradients(zip(gradients, trainable_variables))

                return t_loss, r_loss

        return pretrain

    @tf.function
    def train_step(self, x, y):
        if tf.random.uniform([]) > self.noise_ratio:
            xx = tf.where(tf.random.uniform(x.shape) < self.mask_fraction, 0.0, x)
        else:
            xx = tf.clip_by_value(x + tf.random.normal(x.shape, mean=0, stddev=self.noise_stddev), 0, 1)

        with tf.GradientTape() as tape:
            trainable_variables = []
            r_loss = 0
            for i in range(self.n_layers):
                ae = self.layers[i]
                xx = ae.decoder(ae.encoder(xx, training=True), training=True)

                trainable_variables += ae.trainable_variables
                r_loss += (tf.add_n(ae.encoder.losses) + tf.add_n(ae.decoder.losses))

            # emb = self.predict(x)
            # rp_emb = self.predict(self.rp_features)
            # dis = tf.norm(emb[:, tf.newaxis] - rp_emb, axis=2)
            # pred_ind = tf.argmin(dis, axis=1)
            # a_loss = tf.reduce_mean(tf.norm(y - tf.cast(tf.gather(self.rp_locs, pred_ind), tf.float32), axis=1))
            # a_loss *= self.beta
            a_loss = 0

            r_loss *= self.alpha
            t_loss = self.loss_object(x, xx)
            loss = t_loss + r_loss + a_loss

            gradients = tape.gradient(loss, trainable_variables)
            self.optimizers[-1].apply_gradients(zip(gradients, trainable_variables))

            return t_loss, r_loss, a_loss

    @tf.function
    def predict(self, x):
        xx = x
        for i in range(self.n_layers - 1):
            ae = self.layers[i]
            xx = ae.decoder(ae.encoder(xx))

        xx = self.layers[-1].encoder(xx)
        return xx


class BLEPreprocessor():
    def __init__(self, q=20):
        self.ap_list = None
        self.ap_subset = None
        self.agglo = None
        self.ap_clusters = []
        self.q = q

    def fit(self, X, y, ap_list1, ap_list2, grid_list):
        ## filter mobile ap
        self.ap_list = [x for x in ap_list1 if x in set(ap_list2)]
        print("before feature mapping X:")
        print(X)
        print(y)
        newX = self._feature_mapping(X, self.ap_list)
        print("after feature mapping:")
        print(newX)
        # scale features
        newX = self._feature_scaling(newX)
        print("after feature scaling:")
        print(newX)
        # filter by mutual information
        print("grid_list before feature selection")
        print(grid_list)
        y = [get_closest_grid(loc, grid_list) for loc in y]
        print("newX before feature selection:")
        print(newX)
        print("y before feature selection:")
        print(y)
        newX = self._feature_selection(newX, y)
        print("after feature selection:")
        print(newX)
        # Feature clustering
        #newX = newX[np.sum(newX, axis=1) != 0]
        newX = self._feature_clustering(newX)
        print("after feature_clustering:")
        print(newX)
        return newX

    def preprocess(self, X, ap_list):
        newX = self._feature_mapping(X, ap_list)
        newX = np.where(newX != 0, (newX - self.min_r) / (self.max_r - self.min_r), 0)
        newX = self.agglo.transform(newX)

        return newX

    def save(self, save_dir):
        with open(os.path.join(save_dir, 'preprocessor'), 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(save_dir, 'preprocessor.txt'), 'w') as f:
            f.write(' '.join(self.ap_list) + '\n')
            f.write('%.2f %.2f\n' % (self.min_r, self.max_r))
            f.write(' '.join([str(x) for x in self.ap_clusters]) + '\n')

    def _feature_selection(self, X, y):
        mi = mutual_info_classif(X, y)
        print("mi")
        print(mi)
        print("np.percentile(mi, self.q)")
        print(np.percentile(mi, self.q))
        print(mi > np.percentile(mi, self.q))
        ap_subset = np.flatnonzero(mi > np.percentile(mi, self.q))
        self.ap_list = np.array(self.ap_list)[ap_subset].tolist()
        print("in feature_selection:")
        print("self.ap_list:")
        print(self.ap_list)
        print("ap_subset:")
        print(ap_subset)
        print("X[:,ap_subset]")
        print(X[:,ap_subset])
        return X[:, ap_subset]

    def _feature_clustering(self, X):
        # print(np.isnan(np.corrcoef(X, rowvar=False)).sum())
        self.agglo = FeatureAgglomeration(n_clusters=None, distance_threshold=0.15, affinity='precomputed', linkage='average')
        dist = 1 - np.abs(np.corrcoef(X, rowvar=False))
        print("X")
        print(X)
        self.agglo.fit(dist)
        self.ap_clusters = self.agglo.labels_.tolist()

        return self.agglo.transform(X)

    def _feature_scaling(self, X):
        self.min_r, self.max_r = X.min(), X[np.nonzero(X)].max()
        X = np.where(X != 0, (X - self.min_r) / (self.max_r - self.min_r), 0)

        return X

    def _feature_mapping(self, X, ap_list):
        newX = np.zeros((len(X), len(self.ap_list)))
        for i, ap in enumerate(self.ap_list):
            if ap in ap_list:
                newX[:, i] = X[:, ap_list.index(ap)]
        return newX