import os
import numpy as np
from six.moves import cPickle
import gc
from tensorflow import keras
import utils
from model_zoo import residualbindMPRA

#-----------------------------------------------------------------------------------------

num_trials = 5
batch_size = 100
epochs = 100
save_prefix = 'lentimpra'
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
    
    filepath = '../data/'+cell_type+'_data_with_aleatoric.h5'
    x_train, y_train, x_test, y_test, x_valid, y_valid = utils.load_lentiMPRA_data(filepath)
    N, L, A = x_valid.shape

    for trial in range(num_trials):
        keras.backend.clear_session()
        gc.collect()
        
        savename = os.path.join(results_path, save_prefix + '_' + str(cell_type) + '_' + str(trial))
        print(savename)

        model = residualbindMPRA(input_shape=(L,A))
        model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='mse') 
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                            validation_data=(x_valid, y_valid), callbacks=[es_callback, reduce_lr])
        model.save_weights((savename + '.h5'))

        # evaluate model
        pred = model.predict(x_test, batch_size=batch_size)
        mse, pearsonr, spearmanr = utils.summary_statistics(pred,  y_test, index=0)
        mse2, pearsonr2, spearmanr2 = utils.summary_statistics(pred,  y_test, index=1)
        with open(savename + '.pickle', 'wb') as fout:
            cPickle.dump([mse, pearsonr, spearmanr], fout)
            cPickle.dump([mse2, pearsonr2, spearmanr2], fout)


