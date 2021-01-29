from keras.layers import Dense, Input, LSTM, Embedding, Conv1D, GRU, SpatialDropout1D
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam


from Model.BLCG.RocAucEvaluation import RocAucEvaluation
from Model.BLCG.processing import Procesing4BLCG




class Model4BLCG:

    def __init__(self,path):
        self.path = path


    def checkpoint_model(self, X_valid, Y_valid):
        file_path = "Bi_GRU.hdf5"
        check_point = ModelCheckpoint(file_path, monitor="val_loss", verbose=1,
                                      save_best_only=True, mode="min")
        ra_val = RocAucEvaluation(validation_data=(X_valid, Y_valid), interval=1)
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)

        return [check_point, ra_val, early_stop]

    def build_models(self, X_train_init, X_train, Y_train, X_valid, Y_valid, lr=1e-3, lr_d=0, units=128, dr=0.2, epochs=5, max_len=220, max_features=130000, embed_size=300):
        proces= Procesing4BLCG(self.path)

        print("______________________________Starting build model______________________________")
        inp = Input(shape=(max_len,))
        x = Embedding(max_features, embed_size, weights=[proces.vectorization(X_train_init, embed_size, max_features, max_len)], trainable=False)(inp)
        x1 = SpatialDropout1D(dr)(x)

        x = Bidirectional(GRU(units, return_sequences=True))(x1)
        x = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)

        y = Bidirectional(LSTM(units, return_sequences=True))(x1)
        y = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(y)

        avg_pool1 = GlobalAveragePooling1D()(x)
        max_pool1 = GlobalMaxPooling1D()(x)

        avg_pool2 = GlobalAveragePooling1D()(y)
        max_pool2 = GlobalMaxPooling1D()(y)

        x = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])

        x = Dense(28, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
        history = model.fit(X_train, Y_train, batch_size=128, epochs=epochs, validation_data=(X_valid, Y_valid),
                            verbose=1, callbacks= self.checkpoint_model(X_valid,Y_valid))
        return model