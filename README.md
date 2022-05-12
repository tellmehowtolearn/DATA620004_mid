## YOLO V3训练测试步骤

#### 训练：

首先利用 conda-cpu.yml 或者 conda-gpu.yml 两个文件在Anaconda中创建一个虚拟环境：

```
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov3-tf2-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov3-tf2-gpu
```

在本次实验中采用的是conda-gpu.yml且在windows系统下，需要在https://www.nvidia.com/Download/index.aspx里找到相应的驱动进行更新。

接着下载一个预训练的模型参数： https://pjreddie.com/media/files/yolov3.weights

通过convert.py这个文件加载预训练的模型，在虚拟环境里打开命令行程序并进入存放代码文件的文件夹运行如下代码：

```
python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf
```

那么可以新建出一个文件夹checkpoints并存放初始模型。在本次实验中我们采取VOC2007数据集进行模型的训练，首先在http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html下载好 [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)以便后面的训练。然后将下好的训练验证集解压到data文件夹放进voc2007_raw的文件夹。然后我们用 [tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)去转换数据集，可以根据这个网址提供的教程https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tf-models-install-coco 去完成API的安装。

然后我们执行如下代码来转换数据集：

```
python tools/voc2012.py \
  --data_dir './data/voc2007_raw/VOCdevkit/VOC2007' \
  --split train \
  --output_file ./data/voc2007_train.tfrecord

python tools/voc2012.py \
  --data_dir './data/voc2007_raw/VOCdevkit/VOC2007' \
  --split val \
  --output_file ./data/voc2007_val.tfrecord
```



转化好的数据集已经上传百度云网盘：https://pan.baidu.com/s/1ozTCBl5Mh_gsPYQYiQUluQ?pwd=735m

最后完成了所有准备工作，再来进行模型的训练，由于原预训练模型yolov3有80个类，所以要在20个类上做迁移学习：

```
python train.py \
	--dataset ./data/voc2007_train.tfrecord \
	--val_dataset ./data/voc2007_val.tfrecord \
	--classes ./data/voc2012.names \
	--num_classes 20 \
	--mode fit --transfer darknet \
	--batch_size 16 \
	--epochs 10 \
	--weights ./checkpoints/yolov3.tf \
	--weights_num_classes 80 
```

因为voc2007和voc2012的类别标签一样所以就没有进行更改。

最后生成的模型会放入checkpoints文件夹，每个epoch会存放一个模型为yolov3_train_*.tf。

训练中产生训练和测试的loss会生成一个logs文件夹并进行存放，用tensorboard指令指定其路径即可画出相应的loss曲线。

#### 测试：

我们用最后的模型参数进行图片的测试，用如下代码

```
python detect.py \
	--classes ./data/voc2012.names \
	--num_classes 20 \
	--weights ./checkpoints/yolov3_train_10.tf \
	--image ./data/dog.jpg \
	--output ./data/dog_out.jpg
```

我们进行了三张图片的测试，分别是dog.jpg、bicycle.jpg、car.jpg，并得到相应的结果图片：

dog_out.jpg、bicycle_out.jpg、car_out.jpg。

如果对测试集进行相应的处理使其变成.tfrecord的文件，可以用以下代码进行测试：

```
python detect.py \
	--classes ./data/voc2012.names \
	--num_classes 20 \
	--weights ./checkpoints/yolov3_train_10.tf \
	--tfrecord ./data/voc2007_test.tfrecord
```
