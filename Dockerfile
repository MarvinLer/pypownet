# Use an official Python runtime as a parent image
FROM python:3.6-stretch

MAINTAINER Marvin LEROUSSEAU <marvin.lerousseau@gmail.com>

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y \
    less \
    apt-transport-https \
    software-properties-common
    
# Install octave
RUN apt-get install -y octave && \
    apt-get remove -y software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Git get matpower6.0 and pypownet
RUN git clone https://github.com/MATPOWER/matpower.git && \
    git clone https://github.com/MarvinLer/pypownet.git

# Install pypownet (including necessary packages installation)
WORKDIR pypownet/
RUN cd /pypownet && python setup.py install && cd ..

# Make port 80 available to the world outside this container
EXPOSE 80


# Run the sample experiments when the container launches
CMD ["python3.6", "-m", "pypownet.main"]
