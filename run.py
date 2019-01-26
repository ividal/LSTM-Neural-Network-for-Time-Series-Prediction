import os
import json
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model


def plot_results(predicted_data, true_data, out_path):
    fig = plt.figure(facecolor='white', figsize=(60,10))
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")


def plot_results_multiple(predicted_data, true_data, prediction_len, out_path):
    fig = plt.figure(facecolor='white', figsize=(20,10))
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")


def main():
    config_path = 'config/config.json'
    with open(config_path, 'r') as f:
        configs = json.load(f)
        print("Loaded {}".format(config_path))

    print("\n{}\n".format(configs))

    data_path = configs['data']['filename']
    data_dir = os.path.dirname(data_path)
    dtypes = configs['data'].get('dtypes', None)
    windowed_normalization = configs['data']['normalise']

    data = DataLoader(
            data_path,
            configs['data']['train_test_split'],
            configs['data']['columns'],
            scaler_path=os.path.join(data_dir, "scaler"),
            windowed_normalization=windowed_normalization,
            dtypes=dtypes)

    model = Model()

    if configs['model'].get('load_model'):
        model_path = os.path.join(configs['model']['load_model'])
        print("Loading {}".format(model_path))
        model.load_model(model_path, configs)
        plot_dir = os.path.join(os.path.dirname(model_path), "plots")
        os.makedirs(plot_dir, exist_ok=True)
    else:
        plot_dir = os.path.join(configs['model']['save_dir'], "plots")
        os.makedirs(plot_dir, exist_ok=True)
        model.build_model(configs)
        x, y = data.get_train_data(
                seq_len=configs['data']['sequence_length'],
                windowed_normalization=windowed_normalization
                )

        '''
        # in-memory training
        model.train(
            x,
            y,
            epochs = configs['training']['epochs'],
            batch_size = configs['training']['batch_size'],
            save_dir = configs['model']['save_dir']
        )
        '''
        # out-of-memory generative training
        steps_per_epoch = math.ceil(
                (data.len_train - configs['data']['sequence_length'])
                / configs['training']['batch_size'])
        model.train_generator(
            data_gen=data.generate_train_batch(
                seq_len=configs['data']['sequence_length'],
                batch_size=configs['training']['batch_size'],
                windowed_normalization=windowed_normalization
            ),
            epochs=configs['training']['epochs'],
            batch_size=configs['training']['batch_size'],
            steps_per_epoch=steps_per_epoch,
            save_dir=configs['model']['save_dir']
        )

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        windowed_normalization=windowed_normalization
    )

    predictions_multiple = model.predict_sequences_multiple(x_test, configs['data'][
        'sequence_length'], configs['data']['sequence_length'])
    plot_results_multiple(predictions_multiple, y_test, configs['data']['sequence_length'],
                          out_path=os.path.join(plot_dir, "multiple.png"))

    # predictions_full = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    # plot_results(predictions_full, y_test, os.path.join(plot_dir, "full.png"))
    #
    # predictions_point = model.predict_point_by_point(x_test)
    # plot_results(predictions_point, y_test, os.path.join(plot_dir, "point.png"))


if __name__ == '__main__':
    main()
