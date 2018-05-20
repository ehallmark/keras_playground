import pandas as pd
import keras as k
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Dropout
from keras.optimizers import Adam
csv = pd.read_csv('/Users/inamo/Downloads/daily_stocks.csv', sep=',')

# Examples
technology_sector = csv[csv.sector == 'TECHNOLOGY']
nyse = csv[csv.exchange == 'NYSE']
aapl = technology_sector[technology_sector.ticker == 'AAPL']
nvidia = technology_sector[technology_sector.ticker == 'NVDA']
google = technology_sector[technology_sector.ticker == 'GOOG']
spotify = nyse[nyse.ticker == 'SPOT']  # 'actually in the Consumer Services' sector

#print(aapl[:10])
#print(nvidia[:10])
#print(google[:10])
#print(spotify[:10])

r = 4   # lag
t = 16  # time steps
hidden_layer_size = 128
input_shape = (r, t,)
output_shape = (1,)
batch_size = 128
epochs = 10

# define keras model
x0 = Input(input_shape, name='x')
x = LSTM(hidden_layer_size, activation='tanh', return_sequences=True)(x0)
x = Dropout(0.5)(x)
x = LSTM(hidden_layer_size, activation='tanh', return_sequences=False)(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='tanh')(x)

model = Model(x0, x)

model.compile(optimizer=Adam(0.001, decay=0.00001), loss='mean_squared_error', metrics=['accuracy'])

# define inputs
Xt = None
Y = None
XValT = None
YVal = None

for row in nvidia['adj_close']:
    print(row)

model.fit(Xt, Y, batch_size=batch_size, validation_data=(XValT, YVal), shuffle=True, epochs=epochs)



