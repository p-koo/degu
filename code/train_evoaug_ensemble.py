import os
import numpy as np
from six.moves import cPickle
import gc
import evoaug_tf
from evoaug_tf import evoaug, augment
import utils
from model_zoo import DeepSTARR

#-----------------------------------------------------------------------------------------

downsamples = [1, 0.75, 0.5, 0.25]
num_trials = 5
save_prefix = 'evoaug'
batch_size = 100
epochs = 100
finetune_epochs = 30
results_path = '../results/deepstarr'

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

augment_list = [
    augment.RandomDeletion(delete_min=0, delete_max=20),
    augment.RandomTranslocationBatch(shift_min=0, shift_max=20),
    augment.RandomNoise(noise_mean=0, noise_std=0.2),
    augment.RandomMutation(mutate_frac=0.05)
]

#-----------------------------------------------------------------------------------------


for downsample in downsamples:

    # load dataset
    filepath = '../data/deepstarr_data.h5'
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.load_deepstarr(filepath)
    x_train, y_train = utils.downsample_trainset(x_train, y_train, downsample_frac, seed=12345)
    N, L, A = x_train.shape

    for trial in range(num_trials):
        keras.backend.clear_session()
        gc.collect()

        savename = os.path.join(results_path, save_prefix + '_' + str(downsample) + '_' + str(trial))
        print(savename)

        ##########################################################################################
        # Pre-train with perturbed data
        ##########################################################################################

        model = evoaug.RobustModel(DeepSTARR, input_shape=(L,A), augment_list=augment_list, max_augs_per_seq=2, hard_aug=True)
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer, loss='mse')
        history_aug = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                                validation_data=(x_valid, y_valid), callbacks=[es_callback, reduce_lr])
        model.save_weights(savename+"_aug.h5")
        pred = model.predict(x_test, batch_size=batch_size)
        mse_aug, pearsonr_aug, spearmanr_aug = utils.summary_statistics(pred,  y_test, index=0)
        mse_aug2, pearsonr_aug2, spearmanr_aug2 = utils.summary_statistics(pred,  y_test, index=1)

        ##########################################################################################
        # Fine-tune on original unperturbed data
        ##########################################################################################

        model = evoaug.RobustModel(DeepSTARR, input_shape=(L,A), augment_list=augment_list, max_augs_per_seq=2, hard_aug=True)
        finetune_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(finetune_optimizer, loss='mse')
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


