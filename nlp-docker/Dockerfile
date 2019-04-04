FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
  build-essential \
  wget \
  libevent-dev \
  python3.6 \
  python3.6-dev \
 && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py
RUN ln -s /usr/bin/python3.6 /usr/local/bin/python3

# Install pipenv
ENV PIP_NO_CACHE_DIR=false
RUN pip3 install --upgrade pip pipenv

# Apply the pipenv
RUN mkdir -p /src/
COPY Pipfile* /src/
WORKDIR /src
RUN pipenv --python 3.6 install --system

RUN python3 -m spacy download en
