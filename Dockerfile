########## Install CUDA 11.3 ##########
FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

########## Install basic packages. ##########
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" TZ="Asia/Seoul" apt-get -y install cmake gcc g++ python3.8 python3-pip wget vim tmux gdb curl rsync rename sox ffmpeg openssh-server git nano htop && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

########## Install language setup in Korean ##########
RUN apt-get update && apt-get install locales && \
    locale-gen ko_KR && \
    echo "export LC_ALL=ko_KR.UTF-8" >> /root/.bashrc

########## Install PyTorch ##########
RUN pip install --upgrade pip
RUN pip install numpy scipy tqdm matplotlib sklearn soundfile
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

########## Install evaluation and server files ##########
RUN pip install editdistance flask

########## Install SSH related ##########
RUN echo 'root:ee738' | chpasswd
RUN sed -ri 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
ENTRYPOINT service ssh restart && bash