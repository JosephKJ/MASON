# Unsupervised-Segmentation
Unsupervised Segmentation of Images

### Installation
1. Clone this repository
  ```Shell
  git clone https://github.com/JosephKJ/Unsupervised-Segmentation
  ```
  Let's call the directory as `ROOT`

2. Clone the Caffe repository
  ```Shell
  cd $ROOT
  git clone https://github.com/Microsoft/caffe.git
  ```
  [optional] 
  ```Shell
  cd caffe
  git reset --hard 1a2be8e
  ```
  
3. Get the ImageNet pretrained VGG-16 network(caffemodel only) and place it in `./model` folder. 

caffemodel_url: http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel

