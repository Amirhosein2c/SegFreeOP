
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


class DeepAutoencoder:
    def __init__(self, input_dim, latent_dim=1):
        self.input_dim = input_dim 
        self.latent_dim = latent_dim 
        self.hidden_dim1 = 32
        self.hidden_dim2 = 16
        self.hidden_dim3 = 8 
        self.hidden_dim4 = 4 

        self.input_layer = Input(shape=(self.input_dim,))
        self.hidden_layer1 = Dense(self.hidden_dim1, activation='relu')(self.input_layer)
        self.hidden_layer2 = Dense(self.hidden_dim2, activation='relu')(self.hidden_layer1)
        self.hidden_layer3 = Dense(self.hidden_dim3, activation='relu')(self.hidden_layer2)
        self.hidden_layer4 = Dense(self.hidden_dim4, activation='relu')(self.hidden_layer3)
        self.latent_layer = Dense(self.latent_dim)(self.hidden_layer4)

        self.hidden_layer5 = Dense(self.hidden_dim4, activation='relu')(self.latent_layer)
        self.hidden_layer6 = Dense(self.hidden_dim3, activation='relu')(self.hidden_layer5)
        self.hidden_layer7 = Dense(self.hidden_dim2, activation='relu')(self.hidden_layer6)
        self.hidden_layer8 = Dense(self.hidden_dim1, activation='relu')(self.hidden_layer7)
        self.output_layer = Dense(self.input_dim)(self.hidden_layer8)

        self.autoencoder = Model(self.input_layer, self.output_layer)
        self.encoder = Model(self.input_layer, self.latent_layer)
        self.autoencoder.compile(optimizer='adam', loss='mse')

    def fit(self, X, epochs=50, batch=64, verbose=0):
        self.autoencoder.fit(X, X, epochs=epochs, batch_size=batch, verbose=verbose)

    def encode(self, X):
        encoded_feature = self.encoder.predict(X)
        return encoded_feature


class Autoencoder:
    def __init__(self, input_dim, latent_dim=1):
        self.input_dim = input_dim 
        self.latent_dim = latent_dim 
        
        self.input_layer = Input(shape=(self.input_dim,))
        self.latent_layer = Dense(self.latent_dim)(self.input_layer)
        self.output_layer = Dense(self.input_dim)(self.latent_layer)

        self.autoencoder = Model(self.input_layer, self.output_layer)
        self.encoder = Model(self.input_layer, self.latent_layer)
        self.autoencoder.compile(optimizer='adam', loss='mse')

    def fit(self, X, epochs=250, batch=64, verbose=1):
        self.autoencoder.fit(X, X, epochs=epochs, batch_size=batch, verbose=verbose)

    def encode(self, X):
        encoded_feature = self.encoder.predict(X)
        return encoded_feature
    

class pdAutoEncoder:
    def __init__(self, dataframe):
        
        self.data = dataframe
        self.columns_list = self.data.columns.tolist()
        self.sample_ids = self.data.index.values
        self.X = np.array(self.data.iloc[:, :]) # Exclude first column (sample IDs)
        self.input_dim = self.X.shape[1]
        self.latent_dim = int(self.input_dim/72)
        
        self.input_layer = Input(shape=(self.input_dim,))
        self.latent_layer = Dense(self.latent_dim)(self.input_layer)
        self.output_layer = Dense(self.input_dim)(self.latent_layer)

        self.autoencoder = Model(self.input_layer, self.output_layer)
        self.encoder = Model(self.input_layer, self.latent_layer)
        self.autoencoder.compile(optimizer='adam', loss='mse')
    
    def fit(self, epochs=1000, batch=64, verbose=1):
        self.autoencoder.fit(self.X, self.X, epochs=epochs, batch_size=batch, verbose=verbose)

    def encode(self):
        self.fused_feature_vector = self.encoder.predict(self.X)
        # output_df = pd.DataFrame({'Sample ID': self.sample_ids, 'Fused Features': self.fused_feature_vector})
        output_df = pd.DataFrame(self.fused_feature_vector, index=self.sample_ids)
        return output_df

