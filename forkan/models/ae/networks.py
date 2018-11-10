import sys
import logging

from keras import Model
from keras.layers import Dense, Input

logger = logging.getLogger(__name__)

def create_ae_network(input_shape, latent_dim, network='dense'):

    if network == 'dense':
        # define encoder input layer
        inputs = Input(input_shape)
    
        # prepare encoder
        x = Dense(512, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
    
        # latent variables means and log(standard deviation)
        z = Dense(latent_dim, activation='sigmoid')(x)
    
        # final encoder layer is sampling layer
        encoder = Model(inputs, z, name='encoder')
    
        # define decoder input layer
        de_inputs = Input(shape=(latent_dim,))
    
        # prepare decoder
        x = Dense(128, activation='relu')(de_inputs)
        x = Dense(512, activation='relu')(x)
        x = Dense(input_shape[0], activation='sigmoid')(x)
    
        # decoder restores input in last layer
        decoder = Model(de_inputs, x, name='decoder')
    
        # complete auto encoder
        outputs = decoder(encoder(inputs))
        ae = Model(inputs, outputs, name='full-vae')
    else:
        logger.critical('Network {} does not exist for AE'.format(network))
        sys.exit(1)

    return (inputs, outputs), (encoder, decoder, ae), z