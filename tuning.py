import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import Adam

def build_model(hp):
    # Define search space for hyperparameters
    output_units = len(vocabulary)  # Vocabulary size or number of classes
    
    # Number of units in each GRU layer
    num_units_1 = hp.Int('num_units_1', min_value=32, max_value=256, step=32)
    num_units_2 = hp.Int('num_units_2', min_value=32, max_value=256, step=32)
    num_units_3 = hp.Int('num_units_3', min_value=32, max_value=256, step=32)
    
    # Dropout rate
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    
    # Learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
    
    # Create the model
    model = Sequential()
    model.add(Input(shape=(None, output_units)))  # Input layer

    # Add Bidirectional GRU layers with hyperparameters
    model.add(Bidirectional(GRU(num_units_1, return_sequences=True)))
    model.add(Dropout(dropout_rate))

    model.add(GRU(num_units_2, return_sequences=True))
    model.add(Dropout(dropout_rate))

    model.add(GRU(num_units_3))
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(output_units, activation='softmax'))  # Output layer with softmax activation

    # Compile the model with the given learning rate
    model.compile(
        loss='categorical_crossentropy',  # Assuming a classification task
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    model.summary()
    return model


# Define the tuner using Hyperband
tuner = kt.Hyperband(
    build_model,  # The model function
    objective='val_accuracy',  # Optimize validation accuracy
    max_epochs=10,  # Maximum number of epochs for a single trial
    factor=3,  # The factor to increase the number of trials per stage
    directory='./',  # Directory where the results will be stored
    project_name='lstm_tuning'  # Project name for the experiment
)

# Define the data for training and validation
# Assume you have `inputs` and `targets` for training, and validation data available
# You may need to define validation data or use validation_split

# Run the tuner search for the best hyperparameters
tuner.search(inputs, targets, epochs=10, validation_data=(val_inputs, val_targets))
# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Optionally, you can further train the best model (e.g., using more epochs or different data)
best_model.fit(inputs, targets, epochs=20, validation_data=(val_inputs, val_targets))

