FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ARG UID=1000
ARG GID=1000
ARG USER_NAME
ARG USER_PASSWORD

RUN apt update -y
RUN apt install -y openssh-server
RUN apt install -y rsync
RUN apt install -y git
RUN apt install -y htop
RUN apt install -y tmux
RUN apt install -y unzip
RUN apt install -y vim
RUN apt install -y sudo
RUN apt install -y curl
RUN apt install -y python3-pip

RUN mkdir /var/run/sshd

RUN sed -i 's/\#Port 22/Port 25762/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

RUN groupadd -g $GID -o $USER_NAME
RUN useradd -m -u $UID -g $GID -s /bin/bash $USER_NAME
RUN usermod -a -G sudo $USER_NAME
RUN echo "$USER_NAME:$USER_PASSWORD" | chpasswd
RUN echo "$USER_NAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers
WORKDIR /home/$USER_NAME

USER $USER_NAME
RUN ssh-keygen -q -t rsa -N '' -f ~/.ssh/id_rsa
RUN cat ~/.ssh/id_rsa.pub > ~/.ssh/authorized_keys

ENV HOME=/home/$USER_NAME
ENV PATH="$HOME/.local/bin:$PATH"

RUN python3 -m pip install torch==1.5.1 torchvision

EXPOSE 25762
CMD ["sudo", "/usr/sbin/sshd", "-D"]
