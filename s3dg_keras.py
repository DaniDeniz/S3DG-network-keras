import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


def self_gating(input_tensor, scope, data_format='NDHWC'):

    index_c = data_format.index('C')
    index_d = data_format.index('D')
    index_h = data_format.index('H')
    index_w = data_format.index('W')
    input_shape = input_tensor.get_shape().as_list()
    t = input_shape[index_d]
    w = input_shape[index_w]
    h = input_shape[index_h]
    num_channels = input_shape[index_c]

    spatiotemporal_average = layers.AveragePooling3D(
      (t, w, h),
      strides=(1,1,1),
        padding="valid",
      name=scope + '/self_gating/avg_pool3d')(input_tensor)

    weights = layers.Conv3D(
        num_channels, (1, 1, 1),
        padding="same",
        use_bias=False,
        name=scope + '_conv/self_gating/transformer_W')(spatiotemporal_average)

    tile_multiples = [1, t, w, h]
    tile_multiples.insert(index_c, 1)
 
    weights = tf.keras.backend.tile(weights, tile_multiples)
    weights = layers.Activation("sigmoid")(weights)
    return layers.Multiply()([weights, input_tensor])
    

def conv3d_spatiotemporal(inputs,
                          num_outputs,
                          kernel_size,
                          stride=[1,1,1],
                          padding='same',
                          activation_fn=None,
                          weights_regularizer=l2(1e-7),
                          separable=True,
                          scope=''):
 
    assert len(kernel_size) == 3
    if separable and kernel_size[0] != 1:
        spatial_kernel_size = (1, kernel_size[1], kernel_size[2])
        temporal_kernel_size = (kernel_size[0], 1, 1)
        if isinstance(stride, list) and len(stride) == 3:
            spatial_stride = (1, stride[1], stride[2])
            temporal_stride = (stride[0], 1, 1)
        else:
            spatial_stride = (1, stride, stride)
            temporal_stride = (stride, 1, 1)
        net = layers.Conv3D(
            num_outputs,
            spatial_kernel_size,
            strides=spatial_stride,
            padding=padding,
            use_bias=False,
            kernel_regularizer=weights_regularizer,
            name=scope +"_conv")(inputs)
        net = layers.BatchNormalization(momentum=0.999, scale=False, name=scope+"_bn")(net)
        net = layers.Activation("relu", name = scope)(net)
        net = layers.Conv3D(
            num_outputs,
            temporal_kernel_size,
            strides=temporal_stride,
            padding=padding,
            name=scope + '_conv/t',
            kernel_initializer="glorot_normal")(net)
        net = layers.Activation("relu", name=scope+"/t")(net)
        return net
    else:
        net =  layers.Conv3D(
            num_outputs,
            (kernel_size[0],kernel_size[1],kernel_size[2]),
            strides=(stride[0], stride[1], stride[2]),
            padding=padding,
            activation=activation_fn,
            use_bias=False,
            kernel_regularizer=weights_regularizer,
            name=scope+ "_conv")(inputs)
        net = layers.BatchNormalization(momentum=0.999, scale=False, name= scope +"_bn")(net)
        net = layers.Activation("relu", name=scope)(net)
        return net


def inception_block_v1_3d(inputs,
                          num_outputs_0_0a,
                          num_outputs_1_0a,
                          num_outputs_1_0b,
                          num_outputs_2_0a,
                          num_outputs_2_0b,
                          num_outputs_3_0b,
                          temporal_kernel_size=3,
                          self_gating_fn=None,
                          block=''):

    use_gating = self_gating_fn is not None

    branch_0 = conv3d_spatiotemporal(inputs,
          num_outputs_0_0a, [1, 1, 1], scope='Conv3d_{}_0a_1x1'.format(block))
    if use_gating:
        branch_0 = self_gating_fn(branch_0, scope='Conv3d_{}_0a_1x1'.format(block))
    branch_1 = conv3d_spatiotemporal(inputs,
          num_outputs_1_0a, [1, 1, 1], scope='Conv3d_{}_1a_1x1'.format(block))
    branch_1 = conv3d_spatiotemporal(
          branch_1, num_outputs_1_0b, [temporal_kernel_size, 3, 3],
          scope='Conv3d_{}_1b_3x3'.format(block))
    if use_gating:
        branch_1 = self_gating_fn(branch_1,  scope='Conv3d_{}_1b_3x3'.format(block))
    branch_2 = conv3d_spatiotemporal(inputs,
          num_outputs_2_0a, [1, 1, 1], scope='Conv3d_{}_2a_1x1'.format(block))
    branch_2 = conv3d_spatiotemporal(
          branch_2, num_outputs_2_0b, [temporal_kernel_size, 3, 3],
          scope='Conv3d_{}_2b_3x3'.format(block))
    if use_gating:
        branch_2 = self_gating_fn(branch_2, scope='Conv3d_{}_2b_3x3'.format(block))
    branch_3 = layers.MaxPool3D((3, 3, 3), strides=(1,1,1), padding='same',name='MaxPool3d_{}_3a_3x3'.format(block))(inputs)
    branch_3 = conv3d_spatiotemporal(branch_3,
           num_outputs_3_0b, [1, 1, 1], scope='Conv3d_{}_3b_1x1'.format(block))
    if use_gating:
        branch_3 = self_gating_fn(branch_3, scope='Conv3d_{}_3b_1x1'.format(block))
    
    output = layers.Concatenate(axis=-1, name="Mixed_{}".format(block))([branch_0, branch_1, branch_2, branch_3])
    
    return output


def s3dg_base(inputs,
              first_temporal_kernel_size=3,
              temporal_conv_startat='Conv3d_2c_3x3',
              gating_startat='Conv3d_2c_3x3',
              final_endpoint = 'Mixed_5c',
              data_format='NDHWC'):

    assert data_format in ['NDHWC', 'NCDHW']
    t = 1

    use_gating = False
    self_gating_fn = None

    def gating_fn(inputs, scope):
        return self_gating(inputs, scope, data_format=data_format)

    # batch_size x 32 x 112 x 112 x 64
    end_point = 'Conv3d_1a_7x7'
    if first_temporal_kernel_size not in [1, 3, 5, 7]:
          raise ValueError(
          'first_temporal_kernel_size can only be 1, 3, 5 or 7.')
    # Separable conv is slow when used at first conv layer.
    net = conv3d_spatiotemporal(
        inputs,
        64, [first_temporal_kernel_size, 7, 7],
        stride=[2,2,2],
        separable=False,
        scope=end_point)
    
    # batch_size x 32 x 56 x 56 x 64
    end_point = 'MaxPool2d_2a_3x3'
    net = layers.MaxPool3D(
        (1, 3, 3), strides=(1, 2, 2), padding="same", name=end_point)(net)
    
    # batch_size x 32 x 56 x 56 x 64
    end_point = 'Conv3d_2b_1x1'
    net = conv3d_spatiotemporal(net, 64, [1, 1, 1], scope=end_point)
    # batch_size x 32 x 56 x 56 x 192
    end_point = 'Conv3d_2c_3x3'
    if temporal_conv_startat == end_point:
        t = 3
    if gating_startat == end_point:
        use_gating = True
        self_gating_fn = gating_fn
    net = conv3d_spatiotemporal(net, 192, [t, 3, 3], scope=end_point)
    if use_gating:
        net = self_gating(net, scope=end_point, data_format=data_format)
    
    end_point = 'MaxPool2d_3a_3x3'
    net = layers.MaxPool3D(
        (1, 3, 3), strides=(1, 2, 2), padding="same", name=end_point)(net)

    # batch_size x 32 x 28 x 28 x 256
    end_point = 'Mixed_3b'
    if temporal_conv_startat == end_point:
        t = 3
    if gating_startat == end_point:
        self_gating_fn = gating_fn
    net = inception_block_v1_3d(
        net,
        num_outputs_0_0a=64,
        num_outputs_1_0a=96,
        num_outputs_1_0b=128,
        num_outputs_2_0a=16,
        num_outputs_2_0b=32,
        num_outputs_3_0b=32,
        temporal_kernel_size=t,
        self_gating_fn=self_gating_fn,
        block='3b')
    
    end_point = 'Mixed_3c'
    if temporal_conv_startat == end_point:
        t = 3
    if gating_startat == end_point:
        self_gating_fn = gating_fn
        
    net = inception_block_v1_3d(
        net,
        num_outputs_0_0a=128,
        num_outputs_1_0a=128,
        num_outputs_1_0b=192,
        num_outputs_2_0a=32,
        num_outputs_2_0b=96,
        num_outputs_3_0b=64,
        temporal_kernel_size=t,
        self_gating_fn=self_gating_fn,
        block='3c')
    

    end_point = 'MaxPool3d_4a_3x3'
    net = layers.MaxPool3D(
        (3, 3, 3), strides=(2, 2, 2), padding="same", name=end_point)(net)
    

    # batch_size x 16 x 14 x 14 x 512
    end_point = 'Mixed_4b'
    if temporal_conv_startat == end_point:
        t = 3
    if gating_startat == end_point:
        self_gating_fn = gating_fn
    net = inception_block_v1_3d(
        net,
        num_outputs_0_0a=192,
        num_outputs_1_0a=96,
        num_outputs_1_0b=208,
        num_outputs_2_0a=16,
        num_outputs_2_0b=48,
        num_outputs_3_0b=64,
        temporal_kernel_size=t,
        self_gating_fn=self_gating_fn,
        block='4b')
    

    # batch_size x 16 x 14 x 14 x 512
    end_point = 'Mixed_4c'
    if temporal_conv_startat == end_point:
        t = 3
    if gating_startat == end_point:
        self_gating_fn = gating_fn
    net = inception_block_v1_3d(
        net,
        num_outputs_0_0a=160,
        num_outputs_1_0a=112,
        num_outputs_1_0b=224,
        num_outputs_2_0a=24,
        num_outputs_2_0b=64,
        num_outputs_3_0b=64,
        temporal_kernel_size=t,
        self_gating_fn=self_gating_fn,
        block='4c')

    # batch_size x 16 x 14 x 14 x 512
    end_point = 'Mixed_4d'
    if temporal_conv_startat == end_point:
        t = 3
    if gating_startat == end_point:
        self_gating_fn = gating_fn
    net = inception_block_v1_3d(
        net,
        num_outputs_0_0a=128,
        num_outputs_1_0a=128,
        num_outputs_1_0b=256,
        num_outputs_2_0a=24,
        num_outputs_2_0b=64,
        num_outputs_3_0b=64,
        temporal_kernel_size=t,
        self_gating_fn=self_gating_fn,
        block='4d')

    # batch_size x 16 x 14 x 14 x 528
    end_point = 'Mixed_4e'
    if temporal_conv_startat == end_point:
        t = 3
    if gating_startat == end_point:
        self_gating_fn = gating_fn
    net = inception_block_v1_3d(
        net,
        num_outputs_0_0a=112,
        num_outputs_1_0a=144,
        num_outputs_1_0b=288,
        num_outputs_2_0a=32,
        num_outputs_2_0b=64,
        num_outputs_3_0b=64,
        temporal_kernel_size=t,
        self_gating_fn=self_gating_fn,
        block='4e')

    # batch_size x 16 x 14 x 14 x 832
    end_point = 'Mixed_4f'
    if temporal_conv_startat == end_point:
        t = 3
    if gating_startat == end_point:
        self_gating_fn = gating_fn
    net = inception_block_v1_3d(
        net,
        num_outputs_0_0a=256,
        num_outputs_1_0a=160,
        num_outputs_1_0b=320,
        num_outputs_2_0a=32,
        num_outputs_2_0b=128,
        num_outputs_3_0b=128,
        temporal_kernel_size=t,
        self_gating_fn=self_gating_fn,
        block='4f')
    
    end_point = 'MaxPool3d_5a_2x2'
    net = layers.MaxPool3D(
        (2, 2, 2), strides=(2, 2, 2), padding="same", name=end_point)(net)
    
    # batch_size x 8 x 7 x 7 x 832
    end_point = 'Mixed_5b'
    if temporal_conv_startat == end_point:
        t = 3
    if gating_startat == end_point:
        self_gating_fn = gating_fn
    net = inception_block_v1_3d(
        net,
        num_outputs_0_0a=256,
        num_outputs_1_0a=160,
        num_outputs_1_0b=320,
        num_outputs_2_0a=32,
        num_outputs_2_0b=128,
        num_outputs_3_0b=128,
        temporal_kernel_size=t,
        self_gating_fn=self_gating_fn,
        block='5b')
    
    # batch_size x 8 x 7 x 7 x 1024
    end_point = 'Mixed_5c'
    if temporal_conv_startat == end_point:
        t = 3
    if gating_startat == end_point:
        self_gating_fn = gating_fn
    net = inception_block_v1_3d(
        net,
        num_outputs_0_0a=384,
        num_outputs_1_0a=192,
        num_outputs_1_0b=384,
        num_outputs_2_0a=48,
        num_outputs_2_0b=128,
        num_outputs_3_0b=128,
        temporal_kernel_size=t,
        self_gating_fn=self_gating_fn,
        block='5c')
    
    return net


def s3dg(inputs,
         num_classes=1000,
         first_temporal_kernel_size=3,
         temporal_conv_startat='Conv3d_2c_3x3',
         gating_startat='Conv3d_2c_3x3',
         final_endpoint='Mixed_5c',
         include_top=True,
         dropout_prob=0.5,
         endpoint_logit=False):
    
    net = s3dg_base(inputs, first_temporal_kernel_size, temporal_conv_startat, gating_startat, final_endpoint)
    if include_top:
        # Classification block
        h = int(net.shape[2])
        w = int(net.shape[3])
        x = layers.AveragePooling3D((2, h, w), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(net)
        x = layers.Dropout(dropout_prob)(x)

        x = layers.Conv3D(num_classes, (1, 1, 1),  
                          use_bias=True, 
                          kernel_regularizer=l2(1e-7),
                          name='Conv3d_6a_1x1')(x)
 
        num_frames_remaining = int(x.shape[1])
        x = layers.Reshape((num_frames_remaining, num_classes))(x)

        # logits (raw scores for each class)
        x = layers.Lambda(lambda x: keras.backend.mean(x, axis=1, keepdims=False),
                   output_shape=lambda s: (s[0], s[2]))(x)

        if not endpoint_logit:
            x = layers.Activation('softmax', name='prediction')(x)
    else:
        h = int(net.shape[2])
        w = int(net.shape[3])
        x = layers.AveragePooling3D((2, h, w), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(net)
        
    return Model(inputs, x)



