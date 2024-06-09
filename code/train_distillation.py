import os
import numpy as np
from six.moves import cPickle
import gc
import utils
from model_zoo import DeepSTARR

#-----------------------------------------------------------------------------------------

downsamples = [1, 0.75, 0.5, 0.25]
num_trials = 5
batch_size = 100
epochs = 100
save_prefix = 'deepstarr_distilled'
save_prefix_old = 'deepstarr'
results_path = '../results/deepstarr'

# early stopping callback
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', #'val_aupr',#
                                            patience=10,
                                            verbose=1,
                                            mode='min',
                                            restore_best_weights=True)
# reduce learning rate callback
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                factor=0.2,
                                                patience=3,
                                                min_lr=1e-7,
                                                mode='min',
                                                verbose=1)

#-----------------------------------------------------------------------------------------

for downsample in downsamples:

    # load dataset
    filepath = '../data/deepstarr_data.h5'
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.load_deepstarr(filepath)
    x_train, y_train = utils.downsample_trainset(x_train, y_train, downsample, seed=12345)
    N, L, A = x_train.shape

    ##############################################################################
    # Generate labels from ensemble of models
    ##########################################################################################
    
    y_new = []
    pred = []
    for trial in range(num_trials):

        savename = os.path.join(results_path, save_prefix_old + '_' + str(downsample) + '_' + str(trial) + '.h5')
        model = DeepSTARR(input_shape=(L,A))
        model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='mse') 
        model.load_weights(weight_path)

        # generate predictions
        y_new.append(model.predict(x_train, batch_size=batch_size))
        pred.append(model.predict(x_test, batch_size=batch_size))
    y_train = np.mean(np.array(y_new), axis=0)
    pred = np.mean(np.array(pred), axis=0)

    # run for each set and enhancer type
    pred = model.predict(x_test, batch_size=batch_size)
    mse, pearsonr, spearmanr = utils.summary_statistics(pred,  y_test, index=0)
    mse2, pearsonr2, spearmanr2 = utils.summary_statistics(pred,  y_test, index=1)
    with open(save_prefix_old + '_' + str(downsample) + '_ensemble.pickle', 'wb') as fout:
        cPickle.dump([mse, pearsonr, spearmanr], fout)
        cPickle.dump([mse2, pearsonr2, spearmanr2], fout)

    ##########################################################################################
    # Train distilled models based on ensemble labels
    ##########################################################################################
    
    for trial in range(num_trials):
        keras.backend.clear_session()
        gc.collect()

        savename = os.path.join(results_path, save_prefix + '_' + str(downsample) + '_' + str(trial))
        print(savename)

        model = DeepSTARR(input_shape=(L,A))
        model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='mse') 
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                            validation_data=(x_valid, y_valid), callbacks=[es_callback, reduce_lr])
        model.save_weights(savename + '.h5')

        # evaluate model
        pred = model.predict(x_test, batch_size=batch_size)
        mse_aug, pearsonr_aug, spearmanr_aug = utils.summary_statistics(pred,  y_test, index=0)
        mse_aug2, pearsonr_aug2, spearmanr_aug2 = utils.summary_statistics(pred,  y_test, index=1)
        with open(savename + '.pickle', 'wb') as fout:
            cPickle.dump([mse, pearsonr, spearmanr], fout)
            cPickle.dump([mse2, pearsonr2, spearmanr2], fout)


