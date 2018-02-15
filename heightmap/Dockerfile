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
RUN apt-get update \
    && apt-get install -y \
        git \
        cmake \
        unzip \
        pkg-config \
        libopencv-dev \
        libav-tools  \
        libtiff-dev \
        libjasper-dev \
        libgtk2.0-dev \
        python-numpy \
        python-pycurl \
        libatlas-base-dev \
        webp \
        python-opencv \
        qt5-default \
        zlib1g-dev \
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
        libffi-dev \
        libssl-dev \
        libproj-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev

# Install Open CV - Warning, this takes absolutely forever
RUN mkdir -p ~/opencv cd ~/opencv \
    && wget https://github.com/Itseez/opencv/archive/3.0.0.zip \
    && unzip 3.0.0.zip \
    && rm 3.0.0.zip \
    && mv opencv-3.0.0 OpenCV \
    && cd OpenCV \
    && mkdir build \
    && cd build \
    && cmake \
        -DWITH_QT=ON \
        -DWITH_OPENGL=ON \
        -DFORCE_VTK=ON \
        -DWITH_TBB=ON \
        -DWITH_GDAL=ON \
        -DWITH_XINE=ON \
        -DBUILD_EXAMPLES=ON .. \
    && make -j4 \
    && make install \
    && ldconfig

RUN pip install --upgrade pip \
    && pip install virtualenv virtualenvwrapper numpy \
    && rm -rf ~/.cache/pip \
    && apt-get autoclean && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN cd ~ \
    && git clone https://github.com/Itseez/opencv.git --branch 3.1.0 \
    && git clone https://github.com/Itseez/opencv_contrib.git --branch 3.1.0 \
    && cd /root/opencv \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE  -D CMAKE_INSTALL_PREFIX=/usr/local  -D INSTALL_C_EXAMPLES=OFF  -D INSTALL_PYTHON_EXAMPLES=ON  -D OPENCV_EXTRA_MODULES_PATH=/root/opencv_contrib/modules -D BUILD_EXAMPLES=ON ..

# BUGFIX: https://github.com/opencv/opencv/issues/6050#issuecomment-251684917
RUN echo 'find_package(HDF5)' >> /root/opencv/modules/python/common.cmake \
    && echo 'include_directories(${HDF5_INCLUDE_DIRS})' >> /root/opencv/modules/python/common.cmake

RUN cd ~/opencv/build \
    && make -j $(nproc) \
    && make install \
    && ldconfig \
    && cp ~/opencv/build/lib/cv2.so /usr/local/lib/python2.7/dist-packages/ \
    && rm -rf ~/opencv/build

# https://larrylisky.com/2014/03/03/installing-pcl-on-ubuntu/
RUN apt-get install -y libeigen3-dev \
    && git clone https://github.com/PointCloudLibrary/pcl.git pcl-trunk \
    && cd pcl-trunk \
    && mkdir release \
    && cd release \
    && cmake -DCMAKE_BUILD_TYPE=None -DBUILD_GPU=ON -DBUILD_apps=ON -DBUILD_examples=ON .. \
    && make && make install

RUN apt-get update \
#    && add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl \
#    && apt-get update \
#    && apt-get install libpcl-dev \
    && apt-get install -y libtclap-dev \
    # libpcl-dev
    && pip install requests[security] \
        -U pyOpenSSL \
        ndg-httpsclient \
        pyasn1 \
        cython

RUN pip install git+http://github.com/strawlab/python-pcl.git

#RUN apt-get install libboost-system1.58-dev libboost-system1.58.0

# command to run when starting docker
COPY entrypoint.sh extractor_info.json *.py *.cpp PCLDLL.h CMakeLists.txt /home/extractor/

RUN ldconfig -p | grep libpcl

RUN cd /home/extractor && gcc -o main 2DHeightMap.cpp

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
