## Installation

### Requirements:
- PyTorch 1.0
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- (optional) OpenCV for the webcam demo
- tensorboardX

- Cython
- python3-tk

### Step-by-step installation

```bash
# maskrcnn_benchmark and coco api dependencies
sudo pip3 install torch torchvision
sudo pip3 install ninja yacs cython matplotlib pycocotools tensorboardX Cython

sudo apt-get install python3-tk
sh mmdetection/compile.sh
```

## Start Multi-GPU

### Facebook MASK-RCNN
```bash
export NGPUS=4
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS ./facebook_mrcnn/tools/train_net.py --config-file "path/to/config/file.yaml"
```

### MMLab Detection
- Modify argument **--gpus**

- Use pytorch distributed option
```bash
export NGPUS=4
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS ./mmdetection/tools/train.py --launcher 'pytorch'
```