# LGC-Net

This repo for LGC-Net paper.

## Code
Our implementation is based on Python 3 and [Pytorch](https://pytorch.org/). We
test the code under Ubuntu 18.04, Python 3.8, and Pytorch 1.8. The codebase is licensed under the MIT License.

### Installation & Prerequies
1.  Install the correct version of [Pytorch](http://pytorch.org)
```
pip install --pre torch  -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html
```

2.  Clone this repo and create empty directories
```
git clone https://github.com/huazai665/LGC-Net.git
```

3.  Install the following required Python packages, e.g. with the pip command
```
pip install -r LGC-Net/requirements.txt
```

### Testing
```
cd LGC-Net
python3 main_EUROC.py

# or alternatively
# python3 main_TUMVI.py
```

You can then compare results with the evaluation [toolbox](https://github.com/rpng/open_vins/).

### Training
You can train the method by
uncomment the two lines after # train in the main files. Edit then the
configuration to obtain results with another sets of parameters. 

### Pre_train model
We provide pretraining model in Euroc_results/my/weights.pt and Tum_results/my/weights.pt

### Plot
For EUROC dataset test results
```
python3 plot_result_EUROC.py
```


For TUM-VI dataset test results.
```
python3 plot_result_tum.py
```

