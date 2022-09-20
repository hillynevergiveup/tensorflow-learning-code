# 导包
import matplotlib.pyplot as plt
from model import vgg
import tensorflow as tf
import json
import os
import time
import glob
import random
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'

"""
对训练集进行训练，并用测试集验证
"""

def main():
    # gpu设置
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            exit(-1)

    # 获取训练集、测试集数据的路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
    image_path = os.path.join(data_root, "datasets", "flower_data")
    train_dir = os.path.join(image_path, "train")
    validation_dir = os.path.join(image_path, "val")
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    # 新建保存参数的文件夹
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    # im_height, im_width, batch_size, epochs设置
    im_height = 224
    im_width = 224
    batch_size = 32
    epochs = 10

    # 类别字典，从数据集文件命名获取
    data_class = [cla for cla in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cla))]
    class_num = len(data_class)
    class_dict = dict((value, index) for index, value in enumerate(data_class))

    # 反转字典的key和value
    inverse_dict = dict((val, key) for key, val in class_dict.items())
    # 将字典写入json文件, 先编码成json字符串， 然后写入文件
    json_str = json.dumps(inverse_dict, indent=4) # 将python字典对象编码成json字符串
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)


    # 训练集数据和标签整理
    train_image_list = glob.glob(train_dir + "/*/*.jpg") # 按照指定规则搜索文件，返回所有符合规则的文件列表
    random.shuffle(train_image_list)
    train_num = len(train_image_list)
    assert train_num > 0, "cannot find any .jpg file in {}".format(train_dir)
    train_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in train_image_list] # os.path.sep路径分隔符

    # 测试集数据和标签整理
    val_image_list = glob.glob(validation_dir + "/*/*.jpg")
    val_num = len(val_image_list)
    assert val_num > 0, "cannot find any .jpg file in {}".format(validation_dir)
    val_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in val_image_list]

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    def process_image(img_path, label):
        label = tf.one_hot(label, depth=class_num)
        image = tf.io.read_file(img_path) # 读取文件
        image = tf.image.decode_jpeg(image) # 将jpeg编码的图像解码成uint8张量
        image = tf.image.convert_image_dtype(image, tf.float32) # 转换成tf.float32张量
        image = tf.image.resize(image, [im_height, im_width]) # 调整图片大小
        return image, label

    AUTOTUNE = tf.data.experimental.AUTOTUNE # ???什么意思

    # 加载训练集
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
    train_dataset = train_dataset.shuffle(buffer_size=train_num)\
                                 .map(process_image, num_parallel_calls=AUTOTUNE)\
                                 .repeat().batch(batch_size).prefetch(AUTOTUNE) # repeat()为什么？ prefetch()是什么意思？？

    # 加载测试集
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list, val_label_list))
    val_dataset = val_dataset.map(process_image, num_parallel_calls=AUTOTUNE).repeat().batch(batch_size)

    # 方法1：低层API进行训练
    # 实例化模型
    model = vgg('vgg11', im_height=im_height, im_width=im_width, num_classes=class_num)
    model.summary()

    # 损失值对象
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False) # 分类交叉损失值，前面model中已经计算了softmax，因此logit取False

    # 优化器对象
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    # 训练集损失值、准确率计算方法
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")

    # 测试集损失值、准确率计算
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')


    # @tf.function:修饰符，在函数前加上去，@tf.function 使用静态编译将函数内的代码转换成计算图， 能够提升性能
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def val_step(images, labels):
        predictions = model(images, training=False) # 测试时不使用Dropout，因此training设为False
        v_loss = loss_object(labels, predictions)

        val_loss(v_loss)
        val_accuracy(labels, predictions)

    best_val_loss = float('inf')
    train_step_num = train_num // batch_size
    val_step_num = val_num // batch_size

    for epoch in range(1, epochs+1):
        train_loss.reset_states() # 清空历史信息
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        # 训练集训练
        t1 = time.perf_counter() # 回性能计数器的值（以小数秒为单位）作为浮点数，即具有最高可用分辨率的时钟，以测量短持续时间。
        for index, (images, labels) in enumerate(train_dataset):
            train_step(images, labels)
            if index+1 == train_step_num:
                break
        print("training time:{}/epoch".format(time.perf_counter() - t1)) # 训练一轮的时间

        # 测试集测试
        for index, (images, labels) in enumerate(val_dataset):
            val_step(images, labels)
            if index+1 == val_step_num:
                break

        # 显示训练误差，测试误差
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
        print(template.format(epoch,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              val_loss.result(),
                              val_accuracy.result() * 100
                              ))

        # 保存模型参数
        if val_loss.result() < best_val_loss:
            model.save_weights("./save_weights/myAlex_low_{}.h5".format(epoch), save_format='tf')

    # # 方法2：使用keras高层API进行训练
    # model = AlexNet_v2(num_classes=class_num)
    # model.build((batch_size, im_height, im_width, 3))
    # model.summary()
    #
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    #               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    #               metrics=["accuracy"])
    #
    # callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myAlex_{epoch}.h5',
    #                                                 save_best_only=True,
    #                                                 save_weights_only=True,
    #                                                 monitor='val_loss')]
    #
    # history = model.fit(x=train_dataset,
    #                     steps_per_epoch=train_num // batch_size,
    #                     epochs=epochs,
    #                     validation_data=val_dataset,
    #                     validation_steps=val_num // batch_size,
    #                     callbacks=callbacks)


if __name__ == '__main__':
    main()
