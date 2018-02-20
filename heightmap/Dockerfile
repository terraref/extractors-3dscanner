FROM terraref/terrautils
MAINTAINER Solmaz Hajmohammadi <Solmaz.hajmohammadi@lemanatec.de>


# Various Python and C/build deps
RUN apt-get update \
    && apt-get install -y \
        libtclap-dev \
        libeigen3-dev \
        libpcl-dev \
        libopencv-dev \
        python-opencv


RUN useradd -u 49044 extractor \
    && mkdir -p /home/extractor/sites/ua-mac/raw_data \
    && mkdir -p /home/extractor/sites/ua-mac/Level_1/scanner3DTop \
    && mkdir -p /home/extractor/sites/ua-mac/Level_1/scanner3DTop_mergedlas \
    && mkdir -p /home/extractor/sites/ua-mac/Level_1/scanner3DTop_heightmap \
    && chown -R extractor /home/extractor

WORKDIR /home/extractor

# command to run when starting docker
COPY entrypoint.sh extractor_info.json *.py *.cpp PCLDLL.h CMakeLists.txt /home/extractor/

RUN g++ -o main -I/usr/include/pcl-1.7 -I/usr/include/eigen3 2DHeightMap.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lpcl_common -lpcl_io -lboost_system

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
