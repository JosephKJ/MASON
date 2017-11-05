import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join('/home/joseph/workspace/SDD-RFCN-python/caffe/python')
# caffe_path = osp.join(this_dir, '..', '..', 'caffe', 'python')
add_path(caffe_path)