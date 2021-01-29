import numpy as np

from sklearn.metrics import f1_score
from keras.callbacks import Callback



class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            # y_pred = self.model.predict(self.X_val, verbose=0)
            y_pred = self.model.predict(self.X_val, verbose=0)
            y_pred = np.argmax(y_pred, axis=1)

            f1_score_macro = f1_score(self.y_val, y_pred, average="macro")
            print("\n f1_score_macro - epoch: {:d} - f1_score_macro: {:.6f}".format(epoch + 1, f1_score_macro))
