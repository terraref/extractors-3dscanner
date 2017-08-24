FROM ubuntu:14.04
MAINTAINER Solmaz Hajmohammadi <Solmaz.hajmohammadi@lemanatec.de>

# Install any programs needed
RUN useradd -u 49044 extractor \
    && mkdir -p /home/extractor/sites/ua-mac/raw_data \
    && mkdir -p /home/extractor/sites/ua-mac/Level_1/scanner3DTop \
    && mkdir -p /home/extractor/sites/ua-mac/Level_1/scanner3DTop_mergedlas \
    && mkdir -p /home/extractor/sites/ua-mac/Level_1/scanner3DTop_heightmap \
    && chown -R extractor /home/extractor

# Requirements for terrautils
RUN cd /home/extractor \
    && apt-get -q -y update \
    && apt-get install -y --no-install-recommends build-essential \
        software-properties-common \
        gcc make wget byacc \
        libpng-dev \
        libjpeg8-dev \
        libfreetype6-dev \
        libnetcdf-dev \
        libhdf5-dev \
        libblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        libgdal-dev \
        python-dev \
        python-tk \
    # Requirements for pyclowder
        netcat \
        python \
        python-pip \
        git \
        gcc \
        gfortran \
    && add-apt-repository ppa:ubuntugis/ubuntugis-unstable \
    && apt-get -q -y update \
    && apt-get install -y python-gdal \
    && pip install --upgrade pip setuptools \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Various Python and C/build deps
RUN apt-get update \
    && apt-get install -y \
        wget \
        cmake \
        unzip \
        pkg-config \
        libopencv-dev \
        libav-tools  \
        libjpeg-dev \
        libtiff-dev \
        libjasper-dev \
        libgtk2.0-dev \
        python-numpy \
        python-pycurl \
        libatlas-base-dev \
        webp \
        python-opencv \
        qt5-default \
        #libvtk6-dev \
        zlib1g-dev

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

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y \
        libpng12-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
    && pip install --upgrade pip \
    && pip install virtualenv virtualenvwrapper numpy \
    && rm -rf ~/.cache/pip \
    && apt-get autoclean && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN cd ~ \
    && git clone https://github.com/Itseez/opencv.git \
    && cd opencv \
    && pwd \
    && git checkout 3.1.0 \
    && cd ~ \
    && git clone https://github.com/Itseez/opencv_contrib.git \
    && cd opencv_contrib \
    && git checkout 3.1.0 \
    && cd /root/opencv \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE  -D CMAKE_INSTALL_PREFIX=/usr/local  -D INSTALL_C_EXAMPLES=OFF  -D INSTALL_PYTHON_EXAMPLES=ON  -D OPENCV_EXTRA_MODULES_PATH=/root/opencv_contrib/modules -D BUILD_EXAMPLES=ON ..

RUN cd ~/opencv/build \
    && make -j $(nproc) \
    && make install \
    && ldconfig \
    && cp ~/opencv/build/lib/cv2.so /usr/local/lib/python2.7/dist-packages/

RUN apt-get update \
    && apt-get install -y \
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
    --no-install-recommends

RUN add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl \
    && apt-get update \
    && apt-get install -y libpcl-all libffi-dev libssl-dev

RUN pip install -U requests[security] pyopenssl ndg-httpsclient pyasn1 cython
RUN pip install numpy git+http://github.com/strawlab/python-pcl.git#egg=pcl

# Install pyclowder and terrautils
RUN cd /home/extractor \
    && git clone https://opensource.ncsa.illinois.edu/bitbucket/scm/cats/pyclowder2.git \
    && pip install --upgrade -r pyclowder2/requirements.txt \
    && pip install --upgrade /home/extractor/pyclowder2 \
    && git clone https://github.com/terraref/terrautils.git \
    && cd terrautils && git checkout terrautils_final_prep && cd .. \
    && pip install --upgrade -r terrautils/requirements.txt \
    && pip install --upgrade /home/extractor/terrautils

# command to run when starting docker
COPY entrypoint.sh extractor_info.json *.py *.cpp main PCLDLL.h CMakeLists.txt /home/extractor/

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
