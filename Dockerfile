# Use an official Python runtime as a parent image
FROM python:3.6-stretch

MAINTAINER Marvin LEROUSSEAU <marvin.lerousseau@gmail.com>

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y less apt-transport-https
# Install octave
RUN apt-get install -y software-properties-common
RUN apt-get install -y octave
RUN apt-get remove -y software-properties-common
# cleanup package manager
RUN apt-get autoclean && apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN useradd -ms /bin/bash octave

WORKDIR /l2rpn
# Copy current folder content
ADD . /l2rpn

# Install packages
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80
RUN git clone https://github.com/MATPOWER/matpower.git /matpower6.0

# Run usage_example.py when the container launches
CMD ["python3.6", "-m", "src.usage_example"]