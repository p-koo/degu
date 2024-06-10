import os
import numpy as np
from six.moves import cPickle
import gc
import evoaug_tf
from evoaug_tf import evoaug, augment
import utils
from model_zoo import residualbindMPRA, residualbindMPRA_unc
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#-----------------------------------------------------------------------------------------

num_trials = 5
save_prefix = 'lentimpra_evoaug_distilled'
save_prefix_old = 'lentimpra_evoaug'  

batch_size = 100
epochs = 100
concat = False # add train batch + augmented train batch
results_path = '../results/lentimpra'

augment_list = [
    augment.RandomDeletion(delete_min=0, delete_max=20),
    augment.RandomTranslocationBatch(shift_min=0, shift_max=20),
    augment.RandomNoise(noise_mean=0, noise_std=0.2),
    augment.RandomMutation(mutate_frac=0.05)
]

# early stopping callback
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            verbose=1,
                                            mode='min',
                                            restore_best_weights=True)
# reduce learning rate callback
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                              factor=0.1,
                                              patience=5,
                                              min_lr=1e-7,
                                              mode='min',
                                              verbose=1)

#-----------------------------------------------------------------------------------------


for cell_type in ['HepG2', 'K562']:

    filepath = '../data/'+cell_type+'_data_with_aleatoric.h5'
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

        savename = os.path.join(results_path, save_prefix_old + '_' + str(cell_type) + '_' + str(trial))
        finetune_path = savename+"_finetune.h5"
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

        ##########################################################################################
        # Pre-train with perturbed data
        ##########################################################################################

        model = evoaug.RobustModel(residualbindMPRA_unc, input_shape=(L,A), augment_list=augment_list, 
                                   max_augs_per_seq=2, hard_aug=True)
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer, loss='mse')
        history_aug = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                                validation_data=(x_valid, y_valid), callbacks=[es_callback, reduce_lr])

        save_path = savename+"_aug.h5"
        model.save_weights(save_path)

        # performance metrics
        pred = model.predict(x_test, batch_size=batch_size)
        mse_aug, pearsonr_aug, spearmanr_aug = utils.summary_statistics(pred,  y_test, index=0)
        mse_aug2, pearsonr_aug2, spearmanr_aug2 = utils.summary_statistics(pred,  y_test, index=1)

        ##########################################################################################
        # Fine-tune on original unperturbed data
        ##########################################################################################

        finetune_model = evoaug.RobustModel(residualbindMPRA_unc, input_shape=(L,A), augment_list=augment_list, 
                                   max_augs_per_seq=2, hard_aug=True)
        finetune_optimizer = keras.optimizers.Adam(learning_rate=0.001)
        finetune_model.compile(finetune_optimizer, loss='mse')
        finetune_model.load_weights(save_path)
        finetune_model.finetune_mode()
        history = finetune_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                            validation_data=(x_valid, y_valid), callbacks=[es_callback, reduce_lr])

        finetune_path = savename+"_finetune.h5"
        finetune_model.save_weights(finetune_path)

        # evaluate model
        pred = finetune_model.predict(x_test, batch_size=batch_size)
        mse, pearsonr, spearmanr = utils.summary_statistics(pred,  y_test, index=0)
        mse2, pearsonr2, spearmanr2 = utils.summary_statistics(pred,  y_test, index=1)
        with open(savename + '.pickle', 'wb') as fout:
            cPickle.dump([mse, pearsonr, spearmanr], fout)
            cPickle.dump([mse2, pearsonr2, spearmanr2], fout)
            cPickle.dump([mse_aug, pearsonr_aug, spearmanr_aug], fout)
            cPickle.dump([mse_aug2, pearsonr_aug2, spearmanr_aug2], fout)



