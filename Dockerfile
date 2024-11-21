FROM supervisely/base-py-sdk:6.73.229

RUN pip install opencv-python==4.9.0.80 torch==2.2.1 torchvision==0.17.1

RUN pip3 install git+https://github.com/supervisely-ecosystem/LightGlue.git@main

LABEL python_sdk_version=6.73.229