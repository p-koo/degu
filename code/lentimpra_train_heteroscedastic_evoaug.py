import os, sys
import numpy as np
import gc
from six.moves import cPickle
from tensorflow import keras
import heteroscedastic
import utils
import evoaug_tf
from evoaug_tf import evoaug, augment
from model_zoo import residualbindMPRA

#--------------------------------------------------------------------------------
# set up params

cell_type = sys.argv[1]
loss_name = sys.argv[2]

if loss_name == 'gaussian':
    loss = heteroscedastic.gaussian_nll_loss
elif loss_name == 'laplace':
    loss = heteroscedastic.laplace_nll_loss
elif loss_name == 'cauchy':
    loss = heteroscedastic.cauchy_nll_loss


num_trials = 5
save_prefix = 'lentimpra_evoaug_'+cell_type+'_'+loss_name
batch_size = 100
epochs = 100
results_path = '../results/lentimpra'

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

#--------------------------------------------------------------------------------
# load data

filename = '../data/'+cell_type+'_data_with_aleatoric.h5'
x_train, y_train, x_test, y_test, x_valid, y_valid = utils.load_lentiMPRA_data(filepath)
N, L, A = x_valid.shape

# just use the mean across replicates
y_train = np.expand_dims(y_train[:,0], axis=1)
y_valid = np.expand_dims(y_valid[:,0], axis=1)
y_test = np.expand_dims(y_test[:,0], axis=1)

#--------------------------------------------------------------------------------
# train model

for trial in range(num_trials):
    keras.backend.clear_session()
    gc.collect()

    # set up save_path
    savename = os.path.join(results_path, save_prefix+'_'+str(trial))
    print(savename)

        ##########################################################################################
        # Pre-train with perturbed data
        ##########################################################################################

        model = evoaug.RobustModel(residualbindMPRA, input_shape=(L,A), augment_list=augment_list, max_augs_per_seq=2, hard_aug=True)
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer, loss=loss)
        history_aug = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                                validation_data=(x_valid, y_valid), callbacks=[es_callback, reduce_lr])
        model.save_weights(savename+"_aug.h5")
        pred = model.predict(x_test, batch_size=batch_size)
        mse_aug, pearsonr_aug, spearmanr_aug = utils.summary_statistics(pred,  y_test, index=0)
        mse_aug2, pearsonr_aug2, spearmanr_aug2 = utils.summary_statistics(pred,  y_test, index=1)

        ##########################################################################################
        # Fine-tune on original unperturbed data
        ##########################################################################################

        model = evoaug.RobustModel(residualbindMPRA, input_shape=(L,A), augment_list=augment_list, max_augs_per_seq=2, hard_aug=True)
        finetune_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(finetune_optimizer, loss=loss)
        model.load_weights(savename+"_aug.h5")
        model.finetune_mode()
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                            validation_data=(x_valid, y_valid), callbacks=[es_callback, reduce_lr])
        model.save_weights(savename+"_finetune.h5")

        # evaluate model
        pred = model.predict(x_test, batch_size=batch_size)
        mse, pearsonr, spearmanr = utils.summary_statistics(pred,  y_test, index=0)
        mse2, pearsonr2, spearmanr2 = utils.summary_statistics(pred,  y_test, index=1)
        with open(savename + '.pickle', 'wb') as fout:
            cPickle.dump([mse, pearsonr, spearmanr], fout)
            cPickle.dump([mse2, pearsonr2, spearmanr2], fout)
            cPickle.dump([mse_aug, pearsonr_aug, spearmanr_aug], fout)
            cPickle.dump([mse_aug2, pearsonr_aug2, spearmanr_aug2], fout)


