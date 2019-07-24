FROM nvcr.io/nvidia/pytorch:19.05-py3

COPY requirements.txt /root/requirements.txt
WORKDIR /root
RUN \
pip install --upgrade pip &&\
pip install -r requirements.txt
