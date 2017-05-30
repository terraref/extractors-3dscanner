FROM ubuntu:14.04
MAINTAINER Solmaz Hajmohammadi <Solmaz.hajmohammadi@lemanatec.de>

# Install any programs needed
RUN useradd -u 1000 extractor \
    && apt-get -y update \
    && apt-get install -y -q build-essential git python python-dev python-pip \
    && pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip install pika requests numpy urllib3 enum pyyaml \
    && mkdir /home/extractor \
    && chown -R extractor /home/extractor \
    && cd /home/extractor \
    && git clone https://opensource.ncsa.illinois.edu/bitbucket/scm/cats/pyclowder2.git \
    && cd pyclowder2 \
    && python setup.py install 



RUN mkdir -p /usr/src/app 
WORKDIR /usr/src/app 

# Various Python and C/build deps
RUN apt-get update && apt-get install -y \ 
    wget \
    build-essential \ 
    cmake \ 
    git \
    unzip \ 
    pkg-config \
    python-dev \ 
    python-opencv \ 
    libopencv-dev \ 
    libav-tools  \ 
    libjpeg-dev \ 
    libpng-dev \ 
    libtiff-dev \ 
    libjasper-dev \ 
    libgtk2.0-dev \ 
    python-numpy \ 
    python-pycurl \ 
    libatlas-base-dev \
    gfortran \
    webp \ 
    python-opencv \ 
    qt5-default \
    libvtk6-dev \ 
    zlib1g-dev 

# Install Open CV - Warning, this takes absolutely forever
RUN mkdir -p ~/opencv cd ~/opencv && \
    wget https://github.com/Itseez/opencv/archive/3.0.0.zip && \
    unzip 3.0.0.zip && \
    rm 3.0.0.zip && \
    mv opencv-3.0.0 OpenCV && \
    cd OpenCV && \
    mkdir build && \ 
    cd build && \
    cmake \
    -DWITH_QT=ON \ 
    -DWITH_OPENGL=ON \ 
    -DFORCE_VTK=ON \
    -DWITH_TBB=ON \
    -DWITH_GDAL=ON \
    -DWITH_XINE=ON \
    -DBUILD_EXAMPLES=ON .. && \
    make -j4 && \
    make install && \ 
    ldconfig

RUN \
    apt-get update && apt-get upgrade -y && \
    apt-get install build-essential cmake git pkg-config -y && \
    apt-get install libjpeg8-dev libtiff-dev libjasper-dev libpng12-dev -y && \
    apt-get install libgtk2.0-dev -y && \ 
    apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev -y && \
    apt-get install libatlas-base-dev gfortran -y && \ 
    apt-get install libhdf5-dev -y && \
    apt-get install python-pip -y && \
    pip install --upgrade pip && \
    pip install virtualenv virtualenvwrapper && \
    rm -rf ~/.cache/pip && \
    apt-get install python2.7-dev -y && \
    pip install numpy && \
    apt-get autoclean && apt-get clean && \ 
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*



RUN \
    cd ~ && \
    git clone https://github.com/Itseez/opencv.git && \
    cd opencv && \
    pwd && \
    git checkout 3.1.0 && \ 
    cd ~ && \
    git clone https://github.com/Itseez/opencv_contrib.git && \ 
    cd opencv_contrib && \
    git checkout 3.1.0 && \
    cd /root/opencv && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE  -D CMAKE_INSTALL_PREFIX=/usr/local  -D INSTALL_C_EXAMPLES=OFF  -D INSTALL_PYTHON_EXAMPLES=ON  -D OPENCV_EXTRA_MODULES_PATH=/root/opencv_contrib/modules -D BUILD_EXAMPLES=ON .. 

RUN \
    cd ~/opencv/build && \
    make -j $(nproc) && \
    make install && \
    ldconfig && \
    cp ~/opencv/build/lib/cv2.so /usr/local/lib/python2.7/dist-packages/ 


RUN apt-get update
RUN apt-get install -y \
    wget \
    ca-certificates \
    apparmor \
    aufs-tools \
    automake \
    cmake \
    bash-completion \
    build-essential \
    curl \
    dpkg-sig \
    git \
    iptables \
    libapparmor-dev \
    libcap-dev \
    libsqlite3-dev \
    mercurial \
    parallel \
    python-mock \
    python-pip \
    python-websocket \
    vim \
    --no-install-recommends
#RUN apt-get install -y libpcl-all



RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl
RUN apt-get update && \
    apt-get install -y libpcl-all

RUN apt-get install -y python-pip git
RUN apt-get install -y python-dev
RUN pip install cython
RUN pip install numpy
RUN pip install git+https://github.com/strawlab/python-pcl.git#egg=pcl





# command to run when starting docker
COPY entrypoint.sh extractor_info.json *.py /home/extractor/

USER extractor
ENTRYPOINT ["/home/extractor/entrypoint.sh"]
CMD ["extractor"]

# Setup environment variables. These are passed into the container. You can change
# these to your setup. If RABBITMQ_URI is not set, it will try and use the rabbitmq
# server that is linked into the container. MAIN_SCRIPT is set to the script to be
# executed by entrypoint.sh
ENV RABBITMQ_EXCHANGE="terra" \
    RABBITMQ_VHOST="%2F" \
    RABBITMQ_QUEUE="terra.heightmap" \
    RABBITMQ_URI="" \
MAIN_SCRIPT="terra_heightmap.py"
