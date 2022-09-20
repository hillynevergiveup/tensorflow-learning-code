import os
import json

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from model import vgg

"""
对单张图片进行预测
"""

def main():
    im_height = 224
    im_width = 224

    # 加载图片
    img_path = './tulip.jpg'
    assert os.path.exists(img_path), "file: {} dose not exist.".format(img_path)
    img = Image.open(img_path)

    # 图片预处理：调整大小，像素值缩放到0-1，增加batch维度
    img = img.resize((im_width, im_height))
    plt.imshow(img)
    img = np.array(img) / 255.
    img = (np.expand_dims(img, 0))

    # 从json文件中加载class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: {} dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_dict = json.load(f)

    print(class_dict)


    # 创建模型，使用哪种方式创建的模型，就加载哪种模型训练时保存的参数
    model = vgg('vgg16', im_height, im_width, 5)

    # 加载模型参数
    weight_path = "./save_weights/myAlex_9.h5"
    assert os.path.exists(weight_path), "file: {} dose not exist.".format(weight_path)
    model.load_weights(weight_path)


    # 预测
    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)

    # 打印预测值和概率值
    print_res = "class: {}, prob: {:.3}".format(class_dict[str(predict_class)],
                                                result[predict_class])
    plt.title(print_res)
    # 显示每个类别对应的概率
    for i in range(len(result)):
        print("class: {:10}, prob: {:.3}".format(class_dict[str(i)], result[i]))

    plt.show()

if __name__ == '__main__':
    main()


