from tensorflow import keras 


def train_standard_fun(model, x_train, y_train, validation_data, loss='mse', 
                       max_epochs=100, batch_size=100, initial_lr=0.001, es_patience=10, 
                       lr_decay=0.1, lr_patience=5, **kwargs):
    """Standard model training with early stopping and learning rate scheduling.
    
    Args:
        model: Model to train
        x_train: Training features
        y_train: Training labels
        validation_data: Validation dataset
        loss: Loss function name
        max_epochs: Maximum training epochs
        batch_size: Training batch size
        initial_lr: Initial learning rate
        es_patience: Early stopping patience
        lr_decay: Learning rate decay factor
        lr_patience: Epochs before reducing learning rate
        **kwargs: Additional training parameters
        
    Returns:
        dict: Training history
    """
    optimizer = keras.optimizers.Adam(learning_rate=initial_lr)
    model.compile(optimizer=optimizer, loss=loss)
    
    # Configure callbacks
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=es_patience,
                                              verbose=1,
                                              mode='min',
                                              restore_best_weights=True)
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                factor=lr_decay,
                                                patience=lr_patience,
                                                min_lr=1e-7,
                                                mode='min',
                                                verbose=1)

    # Train model
    history = model.fit(x_train, y_train, epochs=max_epochs, batch_size=batch_size, shuffle=True,
                       validation_data=validation_data, callbacks=[es_callback,reduce_lr])

    return history.history


def train_dynamic_aug_fun(dynamic_model, x_train, y_train, validation_data, loss='mse', 
                         initial_train=train_standard_fun, initial_lr=0.001, 
                         max_finetune_epochs=30, finetune_patience=5, finetune_lr=0.0001, 
                         lr_decay=0.1, lr_patience=3, batch_size=512, **kwargs):
    """Train model with dynamic augmentation followed by finetuning.
    
    Args:
        dynamic_model: Model with augmentation
        x_train: Training features  
        y_train: Training labels
        validation_data: Validation dataset
        loss: Loss function name
        initial_train: Initial training function
        initial_lr: Initial learning rate
        max_finetune_epochs: Maximum finetuning epochs
        finetune_patience: Finetuning early stopping patience
        finetune_lr: Finetuning learning rate
        lr_decay: Learning rate decay factor  
        lr_patience: Patience for learning rate reduction
        batch_size: Training batch size
        **kwargs: Additional parameters
        
    Returns:
        list: [initial_history, finetuning_history]
    """
    # Initial training
    optimizer = keras.optimizers.Adam(learning_rate=initial_lr)
    dynamic_model.compile(optimizer=optimizer, loss=loss)
    history = initial_train(dynamic_model, x_train, y_train, validation_data)

    # Finetuning setup
    finetune_optimizer = keras.optimizers.Adam(learning_rate=finetune_lr)
    dynamic_model.model.compile(finetune_optimizer, loss=loss)
    dynamic_model.finetune_mode()
   
    # Finetuning callbacks
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

