FROM terraref/terrautils
MAINTAINER Solmaz Hajmohammadi <Solmaz.hajmohammadi@lemanatec.de>

# Install any programs needed
RUN useradd -u 49044 extractor \
&& mkdir -p /home/extractor/sites/ua-mac/raw_data \
    && mkdir -p /home/extractor/sites/ua-mac/Level_1/scanner3DTop \
    && mkdir -p /home/extractor/sites/ua-mac/Level_1/scanner3DTop_mergedlas \
    && mkdir -p /home/extractor/sites/ua-mac/Level_1/scanner3DTop_heightmap \
    && chown -R extractor /home/extractor

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
        python-pip \
        libopencv-dev \
        libav-tools  \
        libjpeg-dev \
        libjpeg8-dev \
        libpng-dev \
        libpng12-dev \
        libtiff-dev \
        libjasper-dev \
        libgtk2.0-dev \
        libhdf5-dev \
        numpy \
        python-numpy \
        python-pycurl \
        libatlas-base-dev \
        gfortran \
        webp \
        python-opencv \
        qt5-default \
        libvtk6-dev \
        zlib1g-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        ca-certificates \
        apparmor \
        aufs-tools \
        automake \
        bash-completion \
        curl \
        dpkg-sig \
        iptables \
        libapparmor-dev \
        libcap-dev \
        libsqlite3-dev \
        mercurial \
        parallel \
        python-mock \
        python-websocket \
        vim \
        software-properties-common \
        --no-install-recommends && \
    apt-get autoclean && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl && \
    apt-get update && \
    apt-get install -y libpcl-all

RUN pip install --upgrade pip \
        virtualenv \
        virtualenvwrapper \
        cython \
        numpy \
        git+https://github.com/strawlab/python-pcl.git#egg=pcl && \
    rm -rf ~/.cache/pip

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
RUN cd ~ && \
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
RUN cd ~/opencv/build && \
    make -j $(nproc) && \
    make install && \
    ldconfig && \
    cp ~/opencv/build/lib/cv2.so /usr/local/lib/python2.7/dist-packages/ 

# command to run when starting docker
COPY entrypoint.sh extractor_info.json *.py main *.cpp heightmap PCLDLL.h /home/extractor/

USER extractor
ENTRYPOINT ["/home/extractor/entrypoint.sh"]
CMD ["extractor"]

# Setup environment variables. These are passed into the container. You can change
# these to your setup. If RABBITMQ_URI is not set, it will try and use the rabbitmq
# server that is linked into the container. MAIN_SCRIPT is set to the script to be
# executed by entrypoint.sh
ENV RABBITMQ_EXCHANGE="terra" \
    RABBITMQ_VHOST="%2F" \
    RABBITMQ_QUEUE="terra.3dscanner.heightmap" \
    RABBITMQ_URI="" \
MAIN_SCRIPT="terra_heightmap.py"
