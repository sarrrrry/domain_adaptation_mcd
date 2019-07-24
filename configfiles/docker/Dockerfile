FROM python:3.7.4-slim-stretch AS python-env

COPY requirements.txt /root/requirements.txt
WORKDIR /root
RUN \
pip install --upgrade pip &&\
pip install -r requirements.txt
