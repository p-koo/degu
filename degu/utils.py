import tensorflow as tf 
from tensorflow import keras 
from scipy import stats
from sklearn.metrics import mean_squared_error

#-----------------------------------------------------------------------------
# Useful functions
#-----------------------------------------------------------------------------


def logvar(x, axis=0):
    return tf.math.log(tf.math.reduce_variance(x, axis=axis))

def std(x, axis=0):
    return tf.math.reduce_std(x, axis=axis)


#-----------------------------------------------------------------------------
# Trainging functions
#-----------------------------------------------------------------------------


def train_standard_fun(model, x_train, y_train, validation_data, loss='mse', 
                       max_epochs=5, batch_size=100, initial_lr=0.001, es_patience=10, 
                       lr_decay=0.1, lr_patience=5, **kwargs):

    optimizer = keras.optimizers.Adam(learning_rate=initial_lr)
    model.compile(optimizer=optimizer, loss=loss)
    
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

    return history.history


def train_dynamic_aug_fun(dynamic_model, x_train, y_train, validation_data, loss='mse', 
                          initial_train=train_standard_fun, initial_lr=0.001, 
                          max_finetune_epochs=2,  finetune_patience=10, finetune_lr=0.0001, 
                          lr_decay=0.1, lr_patience=3, batch_size=512, **kwargs):

    
    optimizer = keras.optimizers.Adam(learning_rate=initial_lr)
    dynamic_model.compile(optimizer=optimizer, loss=loss)
    history = initial_train(dynamic_model, x_train, y_train, validation_data)

    # settings for finetuning
    finetune_optimizer = keras.optimizers.Adam(learning_rate=finetune_lr)
    dynamic_model.model.compile(finetune_optimizer, loss=loss)
    dynamic_model.finetune_mode()
   
    # finetune model
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                patience=finetune_patience, 
                                                restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                  factor=lr_decay, 
                                                  patience=lr_patience)
    history2 = dynamic_model.fit(x_train, y_train, 
                                 epochs=max_finetune_epochs, 
                                 batch_size=batch_size, 
                                 shuffle=True, 
                                 validation_data=validation_data, 
                                 callbacks=[es_callback, reduce_lr])

    return [history, history2.history]


#-----------------------------------------------------------------------------
# Evaluation functions
#-----------------------------------------------------------------------------


def eval_regression(pred, y):

    num_tasks = y.shape[1]
    results = []
    for i in range(num_tasks):
        mse = metrics.mean_squared_error(y[:,i], pred[:,i])
        pearsonr = stats.pearsonr(y[:,i], pred[:,i])[0]
        spearmanr = stats.spearmanr(y[:,i], pred[:,i])[0]
        print('Task %d  MSE      = %.4f'%(i, mse))
        print('Task %d  Pearson  = %.4f'%(i, pearsonr))
        print('Task %d  Spearman = %.4f'%(i, spearmanr))
        results.append([mse, pearsonr, spearmanr])
    return results




def eval_classification(pred, y):
    num_tasks = y.shape[1]
    results = []
    for i in range(num_tasks):
        auroc = metrics.roc_auc_score(y[:,i], pred[:,i])
        aupr = metrics.average_precision_score(y[:,i], pred[:,i])  
        f1_score = metrics.f1_score(y[:,i], pred[:,i])  
        print('Task %d  AUROC = %.4f'%(i, auroc))
        print('Task %d  AUPR  = %.4f'%(i, aupr))
        print('Task %d  F1    = %.4f'%(i, f1_score))
        results.append([auroc, aupr, f1_score])
    return results



