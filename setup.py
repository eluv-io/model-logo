from setuptools import setup

setup(
    name='my_package',
    version='0.1',
    packages=['logo'],
    install_requires=[
        'torch==1.9.0',
        'torchvision==0.10.0',
        'scikit-image==0.17.2',
        'easydict==1.9',
        'Pillow==9.4.0',
        'scikit-learn',
        'pandas',
        'wget',
        'docopt',
        'schema',
        'psutil',
        'tqdm',
        'nltk',
        'jiwer',
        'argparse==1.4.0',
        'facenet_pytorch==2.5.2',
        'mxnet-cu101',
        'PyYAML',
        'setproctitle',
        'common_ml @ git+ssh://git@github.com/qluvio/common-ml.git#egg=common_ml',
        'quick_test_py @ git+https://github.com/elv-nickB/quick_test_py.git#egg=quick_test_py'
    ]
)
