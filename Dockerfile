FROM continuumio/miniconda3:latest
MAINTAINER Jiaying Guo (jiaying.guo@ucdconnect.ie)
LABEL Description = "Dockerised CoTV under FLOW and SUMO"

# solve errors about apt-utils
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NOWARNINGS="yes"

# System
RUN apt-get update && \
	apt-get -y upgrade && \
	apt-get install -y vim && \
	pip install -U pip

# Flow dependencies
RUN cd ~ && \
    conda install opencv && \
    pip install tensorflow
    
ENV SUMO_VERSION 1.10.0
ENV SUMO_HOME /opt/sumo
    
# SUMO dependencies
RUN apt-get -qq install \
    wget \
    g++ \
    make \
    cmake \
    swig \
    libproj-dev \
    libgdal-dev \
    libxerces-c-dev \
    libfox-1.6-0 libfox-1.6-dev \
    python3 \
    python3-dev

# Download and extract source code of SUMO
RUN wget http://downloads.sourceforge.net/project/sumo/sumo/version%20$SUMO_VERSION/sumo-src-$SUMO_VERSION.tar.gz
RUN tar xzf sumo-src-$SUMO_VERSION.tar.gz && \
    mv sumo-$SUMO_VERSION $SUMO_HOME && \
    rm sumo-src-$SUMO_VERSION.tar.gz

# Configure and build from source.
RUN cd $SUMO_HOME && \
    mkdir build/cmake-build && \
	cd build/cmake-build && \
	cmake ../.. && \ 
	make && make install
	
# CoTV - project 
RUN git clone https://github.com/Guojyjy/CoTV.git

# Conda env
RUN conda create -n flow python=3.7

# RUN conda init bash
SHELL ["conda", "run", "-n", "flow", "/bin/bash", "-c"]

RUN cd CoTV/flow && \
	pip install -e .
	


