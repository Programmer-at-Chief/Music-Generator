# from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import  plot_model
from keras.models import Sequential
from keras.layers import Input, LSTM, Dropout, Dense
from keras.optimizers import Adam


# Define the model
output_units = 10
num_units = [128, 64,16]
learning_rate = 0.001
loss = 'categorical_crossentropy'

model = Sequential([
    Input(shape=(None, output_units)),
    LSTM(num_units[0], return_sequences=True),
    Dropout(0.2),
    LSTM(num_units[1], return_sequences=True),
    Dropout(0.2),
    LSTM(num_units[2]),
    Dense(output_units, activation='softmax')
])

model.compile(
    loss=loss,
    optimizer=Adam(learning_rate=learning_rate),
    metrics=['accuracy']
)

# Visualize the model
# plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True, dpi=100)
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
