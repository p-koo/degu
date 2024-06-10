import os
import numpy as np
from six.moves import cPickle
import gc
import utils
from model_zoo import residualbindMPRA, residualbindMPRA_unc
from tensorflow import keras

#-----------------------------------------------------------------------------------------

num_trials = 5
batch_size = 100
epochs = 100
save_prefix = 'lentimpra_distilled'
save_prefix_old = 'lentimpra'
results_path = '../results/lentimpra'

# early stopping callback
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
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

for cell_type in ['HepG2', 'K562']:

    filename = '../data/'+cell_type+'_data_with_aleatoric.h5'
    x_train, y_train, x_test, y_test, x_valid, y_valid = utils.load_lentiMPRA_data(filepath)
    N, L, A = x_valid.shape

    ##############################################################################
    # Generate labels from ensemble of models
    ##########################################################################################
    
    y_new = []
    pred = []
    for trial in range(num_trials):
        keras.backend.clear_session()
        gc.collect()

        weight_path = os.path.join(results_path, save_prefix_old + '_' + str(cell_type) + '_' + str(trial) + '.h5')
        model = residualbindMPRA(input_shape=(L,A))
        model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='mse') 
        model.load_weights(weight_path)

        # generate predictions
        y_new.append(model.predict(x_train, batch_size=batch_size))
        pred.append(model.predict(x_test, batch_size=batch_size))
    y_train = np.concatenate([np.mean(np.array(y_new), axis=0), np.std(np.array(y_new), axis=0)], axis=1)
    pred = np.mean(np.array(pred), axis=0)

    # ensemble performance
    mse, pearsonr, spearmanr = utils.summary_statistics(pred,  y_test, index=0)
    mse2, pearsonr2, spearmanr2 = utils.summary_statistics(pred,  y_test, index=1)
    with open(os.path.join(results_path, save_prefix_old + '_' + str(cell_type) + '_ensemble.pickle'), 'wb') as fout:
        cPickle.dump([mse, pearsonr, spearmanr], fout)
        cPickle.dump([mse2, pearsonr2, spearmanr2], fout)

    ##########################################################################################
    # Train distilled models based on ensemble labels
    ##########################################################################################
    
    for trial in range(num_trials):
        keras.backend.clear_session()
        gc.collect()

        savename = os.path.join(results_path, save_prefix + '_' + str(cell_type) + '_' + str(trial))
        print(savename)

        model = residualbindMPRA_unc(input_shape=(L,A))
        model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='mse') 
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                            validation_data=(x_valid, y_valid), callbacks=[es_callback, reduce_lr])
        model.save_weights(savename + '.h5')

        # evaluate model
        pred = model.predict(x_test, batch_size=batch_size)
        mse, pearsonr, spearmanr = utils.summary_statistics(pred,  y_test, index=0)
        mse2, pearsonr2, spearmanr2 = utils.summary_statistics(pred,  y_test, index=1)
        with open(savename + '.pickle', 'wb') as fout:
            cPickle.dump([mse, pearsonr, spearmanr], fout)
            cPickle.dump([mse2, pearsonr2, spearmanr2], fout)


