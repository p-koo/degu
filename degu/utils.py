
#-----------------------------------------------------------------------------
# Useful functions
#-----------------------------------------------------------------------------

def log_var(x, axis=0):
    return tf.math.log(tf.math.reduce_variance(x, axis=axis))

def std(x, axis=0):
    return tf.math.reduce_std(x, axis=axis)

def log_var_np(x, axis=0):
    return np.log(np.var(x, axis=axis))

def std_np(x, axis=0):
    return np.std(x, axis=axis)


#-----------------------------------------------------------------------------
# Trainging functions
#-----------------------------------------------------------------------------


def train_fun(model, x_train, y_train, validation_data, save_prefix,
                         max_epochs=100, batch_size=100, es_patience=10, 
                         lr_decay=0.1, lr_patience=5):

    # early stopping callback
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=es_patience,
                                                verbose=1,
                                                mode='min',
                                                restore_best_weights=True)
    # reduce learning rate callback
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  factor=lr_decay,
                                                  patience=lr_patience,
                                                  min_lr=1e-7,
                                                  mode='min',
                                                  verbose=1)

    # train model
    history = model.fit(x_train, y_train, epochs=max_epochs, batch_size=batch_size, shuffle=True,
                        validation_data=validation_data, callbacks=[es_callback,reduce_lr])

    # save weights
    save_path = save_prefix+'.weights.h5'
    model.save_weights(save_path)

    return history.history


def setup_dynamic_student_model(model_fun, input_shape, augment_list,
                                hard_aug=True, max_augs_per_seq=2): 
    
    model = evoaug.RobustModel(model_fun, 
                               input_shape=input_shape, 
                               augment_list=augment_list,
                               max_augs_per_seq=max_augs_per_seq, 
                               hard_aug=hard_aug)
    return model


def train_dynamic_aug_fun(model, x_train, y_train, validation_data, save_prefix, 
                          finetune_lr=0.0001, loss='mse'):

    save_path = save_prefix+'_aug.weights.h5'
    deepstarr_train_fun(model, x_train, y_train, validation_data, save_path)

    # settings for finetuning
    finetune_optimizer = keras.optimizers.Adam(learning_rate=finetune_lr)
    model.compile(finetune_optimizer, loss=loss)
    model.finetune_mode()

    # finetune model
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=finetune_patience, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_decay, patience=lr_patience)
    history2 = model.fit(x_train, y_train, epochs=max_finetune_epochs, batch_size=batch_size, shuffle=True,
                         validation_data=(x_valid, y_valid), callbacks=[es_callback, reduce_lr])

    # save finetune weights
    model_path = save_path+'_finetune.weights.h5'
    model.save_weights(model_path)

    return model, model_path, [history.history, history2.history]


def eval_regression(pred, y):

    num_tasks = y.shape[1]
    results = []
    for i in range(num_tasks):
        mse = mean_squared_error(y[:,i], pred[:,i])
        pearsonr = stats.pearsonr(y[:,i], pred[:,i])[0]
        spearmanr = stats.spearmanr(y[:,i], pred[:,i])[0]
        print('Task %d  MSE = %.4f'%(i, mse))
        print('Task %d  PCC = %.4f'%(i, pearsonr))
        print('Task %d  SCC = %.4f'%(i, spearmanr))
        results.append([mse, pearsonr, spearmanr])
    return results



