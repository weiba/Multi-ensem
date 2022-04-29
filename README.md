# Multi-ensem
The data and sourcecode of the paper entitled "Dai, Wei, Bingxi Chen, Wei Peng, Xia Li, Jiancheng Zhong, and Jianxin Wang. "A Novel Multi-Ensemble Method for Identifying Essential Proteins." Journal of Computational Biology 28, no. 7 (2021): 637-649."

Multi-ensemble: a novel ensemble method to improve identification of essential proteins.

Usage：
Step 1.Installation
Installation has been tested in Linux with Python 2.7.13.
Since the method is written in python 2.7.13, python 2.7.13 with the pip tool must be installed first. Multi-ensemble uses the following dependencies: numpy, pandas, scikit-learn, tensorflow version=1.2.0 You can install these packages first, by the following commands:

pip install pandas
pip install numpy
pip install scikit-learn
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.0-cp27-none-linux_x86_64.whl

Step 2.Modify the file path
There is an operation to read data in the program, so change to your own data path before running.

Step 3.Train and test Multi-ensemble
For Yeast data, run：
python tri26_sigma.py

For E.coli data, run:
python tri8_sigma.py
