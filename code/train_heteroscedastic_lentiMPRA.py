import os, sys
import h5py
import gc
import numpy as np
from six.moves import cPickle
import tensorflow as tf
from tensorflow import keras
import keras.layers as kl
from scipy import stats
from sklearn.metrics import mean_squared_error

#--------------------------------------------------------------------------------

def gaussian_nll_loss(y_true, y_pred):
    mean = tf.expand_dims(y_pred[:,0], axis=1)
    log_variance = tf.expand_dims(y_pred[:,1], axis=1)

    # Calculate the negative log-likelihood
    mse = keras.losses.mean_squared_error(y_true, mean)
    variance = tf.exp(log_variance)
    nll = 0.5 * (tf.math.log(2 * np.pi * variance) + mse / variance)

    # Return the average NLL across the batch
    return tf.reduce_mean(nll)
    

def laplace_nll_loss(y_true, y_pred):
    mu = tf.expand_dims(y_pred[:,0], axis=1)
    log_b = tf.expand_dims(y_pred[:,1], axis=1)

    # Calculate the absolute error
    abs_error = tf.abs(y_true - mu)

    # Calculate the negative log-likelihood
    b = tf.exp(log_b)
    nll = abs_error / b + log_b + tf.math.log(2.0)

    # Return the average NLL across the batch
    return tf.reduce_mean(nll)


def cauchy_nll_loss(y_true, y_pred):
    mu = tf.expand_dims(y_pred[:,0], axis=1)
    log_b = tf.expand_dims(y_pred[:,1], axis=1)

    # Calculate the negative log-likelihood
    b = tf.exp(log_b)
    nll = tf.math.log(np.pi * b) + tf.math.log(1 + tf.square((y_true - mu) / b))

    # Return the average NLL across the batch
    return tf.reduce_mean(nll)



def gaussian_confidence_interval(pred, alpha=0.05):
    mean = pred[:,0]
    std_dev = np.sqrt(np.exp(pred[:,1]))
    z_score = stats.norm.ppf(1 - alpha / 2)
    lower_bound = mean - z_score * std_dev
    upper_bound = mean + z_score * std_dev
    return lower_bound, upper_bound


def laplace_confidence_interval(pred, alpha=0.05):
    mu = pred[:,0]
    b = np.exp(pred[:,1])
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2
    lower_bound = mu - b * np.log(1 / lower_quantile)
    upper_bound = mu + b * np.log(1 / (1 - upper_quantile))
    return lower_bound, upper_bound


def cauchy_confidence_interval(pred, alpha=0.05):
    mu = pred[:,0]
    b = np.exp(pred[:,1])
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2
    lower_bound = mu - b * np.tan(np.pi * (upper_quantile - 0.5))
    upper_bound = mu + b * np.tan(np.pi * (0.5 - lower_quantile))
    return lower_bound, upper_bound


def load_lentiMPRA_data(file):
    '''
    load Train/Test/Val lentiMPRA data
    '''
    data = h5py.File(file, 'r')

    # train
    X_train = np.array(data['Train']['X'])
    y_train = np.array(data['Train']['y'])

    # test
    X_test = np.array(data['Test']['X'])
    y_test = np.array(data['Test']['y'])

    # val
    X_val = np.array(data['Val']['X'])
    y_val = np.array(data['Val']['y'])

    return X_train, y_train, X_test, y_test, X_val, y_val


def summary_statistics(pred, Y, task, index):
    mse = mean_squared_error(Y[:,index], pred[:,index])
    pcc = stats.pearsonr(Y[:,index], pred[:,index])[0]
    scc = stats.spearmanr(Y[:,index], pred[:,index])[0]
    print(' MSE ' + task + ' = ' + str("{0:0.3f}".format(mse)))
    print(' PCC ' + task + ' = ' + str("{0:0.3f}".format(pcc)))
    print(' SCC ' + task + ' = ' + str("{0:0.3f}".format(scc)))
    return mse, pcc, scc


def residualbindMPRA(input_shape):
    '''
    CNN for predicting lentiMPRA data
    if aleatoric=True, predict aleatoric uncertainty
    if epistemic=True, predict epistemic uncertainty 
    '''

    def residual_block(input_layer, filter_size, activation='relu', dilated=5):
        '''
        define residual block for CNN
        '''
        factor = []
        base = 2
        for i in range(dilated):
            factor.append(base**i)
        num_filters = input_layer.shape.as_list()[-1]

        nn = kl.Conv1D(filters=num_filters,
                                        kernel_size=filter_size,
                                        activation=None,
                                        use_bias=False,
                                        padding='same',
                                        dilation_rate=1, 
                                        )(input_layer)
        nn = kl.BatchNormalization()(nn)
        for f in factor:
            nn = kl.Activation('relu')(nn)
            nn = kl.Dropout(0.1)(nn)
            nn = kl.Conv1D(filters=num_filters,
                                            kernel_size=filter_size,
                                            activation=None,
                                            use_bias=False,
                                            padding='same',
                                            dilation_rate=f,
                                            )(nn)
            nn = kl.BatchNormalization()(nn)
        nn = kl.add([input_layer, nn])
        return kl.Activation(activation)(nn)

    inputs = kl.Input(shape=input_shape)
    x = kl.Conv1D(196, kernel_size=19, padding='same')(inputs)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('silu')(x)
    x = kl.Dropout(0.2)(x)
    x = residual_block(x, 3, activation='silu', dilated=5)
    x = kl.Dropout(0.2)(x)
    x = kl.MaxPooling1D(5)(x) 

    x = kl.Conv1D(256, kernel_size=7, padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('silu')(x)
    x = kl.Dropout(0.2)(x)
    x = kl.MaxPooling1D(5)(x) 

    x = kl.Dense(256)(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('silu')(x)
    x = kl.Dropout(0.5)(x)

    x = kl.GlobalAveragePooling1D()(x)
    x = kl.Flatten()(x)

    x = kl.Dense(256)(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('silu')(x)
    x = kl.Dropout(0.5)(x)

    outputs = kl.Dense(2, activation='linear')(x)

    return keras.Model(inputs=inputs, outputs=outputs)

#--------------------------------------------------------------------------------
# set up params

cell_type = sys.argv[1]
loss_name = sys.argv[2]
num_trials = int(sys.argv[3])

if loss_name == 'gaussian':
    loss = gaussian_nll_loss
elif loss_name == 'laplace':
    loss = laplace_nll_loss
elif loss_name == 'cauchy':
    loss = cauchy_nll_loss

base_path = 'lentiMPRA_'+cell_type+'_'+loss_name

#--------------------------------------------------------------------------------
# load data

filename = cell_type+'_data_with_aleatoric.h5'
x_train, y_train, x_test, y_test, x_valid, y_valid = load_lentiMPRA_data(filename)
N, L, A = x_valid.shape

#--------------------------------------------------------------------------------
# train model

for trial in range(num_trials):
    keras.backend.clear_session()
    gc.collect()

    # set up save_path
    save_path = os.path.join(base_path+'_'+str(trial))
    print(save_path)

    # load and compile model
    model = residualbindMPRA((L,A))
    model.compile(keras.optimizers.Adam(learning_rate=0.001), loss=loss)

    # early stopping callback
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                patience=10,
                                                verbose=1,
                                                mode='min',
                                                restore_best_weights=True)
    # reduce learning rate callback
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    factor=0.2,
                                                    patience=5,
                                                    min_lr=1e-7,
                                                    mode='min',
                                                    verbose=1)
    # train model
    history = model.fit(x_train, y_train,
                        epochs=100,
                        batch_size=128,
                        shuffle=True,
                        validation_data=(x_valid, y_valid),
                        callbacks=[es_callback, reduce_lr])

    # evaluate results
    pred = model.predict(x_test, batch_size=512)
    mse, pcc, scc    = summary_statistics(pred, y_test, task='mean', index=0)
    if loss_name == 'gaussian':
        mse2, pcc2, scc2 = summary_statistics(np.sqrt(np.exp(pred)), y_test, task='std', index=1)
    elif loss_name == 'laplace':
        mse2, pcc2, scc2 = summary_statistics(np.exp(pred), y_test, task='std', index=1)
    elif loss_name == 'cauchy':
        mse2, pcc2, scc2 = summary_statistics(np.exp(pred), y_test, task='std', index=1)

    # save model weights
    model.save_weights(save_path+'.h5')

    # save results in pickle
    with open(save_path+'.pickle', 'wb') as fout:
        cPickle.dump([mse, pcc, scc], fout)
        cPickle.dump([mse2, pcc2, scc2], fout)
        cPickle.dump(pred, fout)
        cPickle.dump(history.history, fout)


