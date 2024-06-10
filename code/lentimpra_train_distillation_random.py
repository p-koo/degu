import os
import numpy as np
from six.moves import cPickle
import gc
from tensorflow import keras
import evoaug_tf
from evoaug_tf import evoaug, augment
import evoaug_custom
import utils
from model_zoo import residualbindMPRA, residualbindMPRA_unc
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#-----------------------------------------------------------------------------------------

num_trials = 5
save_prefix = 'lentimprea_distilled_random'
save_prefix_old = 'lentimpra' 

batch_size = 100
epochs = 100
finetune_epochs = 30
concat = False # add train batch + augmented train batch
results_path = '../results/lentimpra'

augment_list = [
    augment.RandomMutation(mutate_frac=0.7)
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

    filename = '../data/'+cell_type+'_data_with_aleatoric.h5'
    x_train, y_train, x_test, y_test, x_valid, y_valid = utils.load_lentiMPRA_data(filepath)
    N, L, A = x_valid.shape

    ##########################################################################################
    # Load ensemble of models
    ##########################################################################################

    ensemble_models = []
    for trial in range(num_trials):
        weight_path = os.path.join(results_path, save_prefix_old + '_' + str(cell_type) + '_' + str(trial) + '.h5')
        model = residualbindMPRA(input_shape=(L,A))
        model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='mse') 
        model.load_weights(weight_path)
        ensemble_models.append(model)


    for trial in range(num_trials):
        keras.backend.clear_session()
        gc.collect()

        savename = os.path.join(results_path, save_prefix + '_' + str(cell_type) + '_' + str(trial))
        print(savename)

        ##########################################################################################
        # Pre-train with perturbed data
        ##########################################################################################

        model = evoaug_custom.AugModel(residualbindMPRA_unc, ensemble_models, input_shape=(L,A), augment_list=augment_list, 
                                max_augs_per_seq=2, hard_aug=True, concat=concat)
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

        finetune_model = evoaug.RobustModel(residualbindMPRA_unc, input_shape=(L,A), augment_list=[], 
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



