from tensorflow.keras import layers, Model, Sequential
"""
真是太简洁了！
"""

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config':{
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

# cfgs网络结构设置
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def make_feature(cfg):
    # 根据cfg配置参数搭建卷积层、池化层
    feature_layers = []
    for v in cfg:
        if v == 'M':
            feature_layers.append(layers.MaxPool2D(pool_size=2, strides=2))
        else:
            conv2d = layers.Conv2D(v, kernel_size=3, padding='SAME', activation='relu', kernel_initializer=CONV_KERNEL_INITIALIZER)
            feature_layers.append(conv2d)
    return Sequential(feature_layers, name='feature')

def VGG(feature, im_height=224, im_width=224, num_classes=1000):
    # 构建模型
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype='float32')
    x = feature(input_image)
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu', kernel_initializer=DENSE_KERNEL_INITIALIZER)(x) # 原文是4096
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu', kernel_initializer=DENSE_KERNEL_INITIALIZER)(x) # 原文是4096
    x = layers.Dense(num_classes, kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)
    output = layers.Softmax()(x)
    model = Model(inputs=input_image, outputs=output)
    return model

def vgg(model_name='vgg', im_height=224, im_width=224, num_classes=1000):
    assert model_name in cfgs.keys(), "not support model {}".format(model_name)
    cfg = cfgs[model_name]
    feature = make_feature(cfg)
    model = VGG(feature, im_height=im_height, im_width=im_width, num_classes=num_classes)
    return model