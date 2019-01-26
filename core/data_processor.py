import logging
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

logger = logging.getLogger(__name__)


class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols, scaler_path, windowed_normalization,
                 dtypes=None):
        dataframe = pd.read_csv(filename, dtype=dtypes)
        i_split = int(len(dataframe) * split)
        selected_cols = dataframe[cols]

        if not windowed_normalization:
            # Normalize and save scaler
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logging.info("Reading scaler from {}".format(scaler_path))
            else:
                train_df = selected_cols.iloc[:i_split]
                self.create_scaler(train_df[cols])
                joblib.dump(self.scaler, scaler_path)
                logging.info("Saved scaler to {}".format(scaler_path))

            logging.info("Pre-norm: \nmin \n{}, \n\nmax \n{}, \n\nmean \n{}".format(
                    selected_cols.iloc[:,0:5].min(),
                    selected_cols.iloc[:,0:5].max(),
                    selected_cols.iloc[:,0:5].mean()))

            # normalized_selected = pd.DataFrame(columns=cols)
            selected_cols[cols] = pd.DataFrame(self.scaler.transform(selected_cols[cols]))

            logging.info("Post-norm: \nmin \n{}, \n\nmax \n{}, \n\nmean \n{}".format(
                    selected_cols.iloc[:,0:5].min(),
                    selected_cols.iloc[:,0:5].max(),
                    selected_cols.iloc[:,0:5].mean()))

        self.data_train = selected_cols.values[:i_split]
        self.data_test  = selected_cols.values[i_split:]

        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, windowed_normalization):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        if windowed_normalization:
            data_windows = self.normalise_windows(data_windows, single_window=False)

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x,y

    def get_train_data(self, seq_len, windowed_normalization):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, windowed_normalization)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, windowed_normalization):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, windowed_normalization)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, windowed_normalization):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        if windowed_normalization:
            window = self.normalise_windows(window, single_window=True)[0]
        x = window[:-1]
        y = window[-1, [0]]
        return x, y


    def create_scaler(self, data):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler = self.scaler.fit(data)


    def unscale_set(self, normalized):
        data = self.scaler.inverse_transform(normalized)
        return data


    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)