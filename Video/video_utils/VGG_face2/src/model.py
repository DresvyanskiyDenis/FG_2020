import tensorflow as tf

from Video.video_utils.VGG_face2.src import resnet

global weight_decay
weight_decay = 1e-4


def Vggface2_ResNet50(input_dim=(224, 224, 3), nb_classes=8631, mode='train'):
    # inputs are of size 224 x 224 x 3
    inputs = tf.keras.layers.Input(shape=input_dim, name='base_input')
    x = resnet.resnet50_backend(inputs)

    # AvgPooling
    x = tf.keras.layers.AveragePooling2D((7, 7), name='avg_pool')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu', name='dim_proj')(x)

    if mode == 'train':
        y = tf.keras.layers.Dense(nb_classes, activation='softmax',
                               use_bias=False, trainable=True,
                               kernel_initializer='orthogonal',
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                               name='classifier_low_dim')(x)
    else:
        y = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, 1))(x)

    # Compile
    model = tf.keras.models.Model(inputs=inputs, outputs=y)

    return model

