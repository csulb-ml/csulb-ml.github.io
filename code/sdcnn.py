from keras.models import Model
from keras.layers import Input, Conv2D, Reshape, core, Dropout, Add, SeparableConv2D, BatchNormalization
from keras.optimizers import Adam

# Smoothed Dilated CNN (234) with Shared and Separable Convolution (SSC) included
def sdcnn(patch_height, patch_width, n_ch):
    print('********************SS_DCNN********************')
    inputs = Input(shape=(patch_height, patch_width, n_ch))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(inputs)
    dilblock1 = dilation_block(conv1, 32, (3, 3), _dilation_rate=(2, 2))
    
    conv2 = Conv2D(64, (1, 1), activation='relu', padding='same', data_format='channels_last')(dilblock1)
    dilblock2 = dilation_block(conv2, 64, (3, 3), _dilation_rate=(3, 3))
    
    conv3 = Conv2D(128, (1, 1), activation='relu', padding='same', data_format='channels_last')(dilblock2)
    dilblock3 = dilation_block(conv3, 128, (3, 3), _dilation_rate=(4, 4), ssc=True) # Added SSC to this dilation block
    
    final = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_last')(dilblock3)
    final = core.Reshape((patch_height*patch_width, 2))(final)
    final = core.Activation('softmax')(final)

    model = Model(inputs=inputs, outputs=final)
    print(model.summary())

    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def dilation_block(_inputs, _filters, _kernel_size, _strides=1, _dilation_rate=1, ssc=False):
    if ssc == True:
        block_1 = SeparableConv2D(_filters, _kernel_size, activation='relu', padding='same', data_format='channels_last', depth_multiplier=1, use_bias=False)(_inputs)
        block_1 = BatchNormalization()(block_1)
        block_1 = Conv2D(_filters, _kernel_size, activation='relu', padding='same', data_format='channels_last', dilation_rate=_dilation_rate)(block_1)
    else:
        block_1 = Conv2D(_filters, _kernel_size, activation='relu', padding='same', data_format='channels_last', dilation_rate=_dilation_rate)(_inputs)
    block_1 = BatchNormalization()(block_1)
    if ssc == True:
        block_1 = Conv2D(_filters, _kernel_size, activation='relu', padding='same', data_format='channels_last', dilation_rate=_dilation_rate)(block_1)
    block_1 = Conv2D(_filters, _kernel_size, activation='relu', padding='same', data_format='channels_last', dilation_rate=_dilation_rate)(block_1)
    block_1 = BatchNormalization()(block_1)
    
    merge_1 = Add()([_inputs, block_1])
    
    if ssc == True:
        block_2 = SeparableConv2D(_filters, _kernel_size, activation='relu', padding='same', data_format='channels_last', depth_multiplier=1, use_bias=False)(merge_1)
        block_1 = BatchNormalization()(block_1)
    block_2 = Conv2D(_filters, _kernel_size, activation='relu', padding='same', data_format='channels_last', dilation_rate=_dilation_rate)(merge_1)
    block_2 = BatchNormalization()(block_2)
    if ssc == True:
        block_2 = SeparableConv2D(_filters, _kernel_size, activation='relu', padding='same', data_format='channels_last', depth_multiplier=1, use_bias=False)(block_2)
    block_2 = Conv2D(_filters, _kernel_size, activation='relu', padding='same', data_format='channels_last', dilation_rate=_dilation_rate)(block_2)
    block_2 = BatchNormalization()(block_2)
    
    merge_2 = Add()([merge_1, block_2])

    return merge_2
    