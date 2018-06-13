FROM terraref/terrautils
MAINTAINER Max Burnette <mburnet2@illinois.edu>

# Install any programs needed
RUN useradd -u 49044 extractor \
    && mkdir -p /home/extractor/sites/ua-mac/raw_data \
    && mkdir -p /home/extractor/sites/ua-mac/Level_1/scanner3DTop \
    && mkdir -p /home/extractor/sites/ua-mac/Level_1/scanner3DTop_mergedlas \
    && mkdir -p /home/extractor/sites/ua-mac/Level_1/scanner3DTop_heightmap \
    && chown -R extractor /home/extractor

# Install laser3d science package
RUN pip install laspy plyfile terraref-laser3d

# Install Wine
RUN dpkg --add-architecture i386 \
    && wget -nc https://dl.winehq.org/wine-builds/Release.key \
    && apt-key add Release.key \
    && apt-add-repository https://dl.winehq.org/wine-builds/ubuntu/ \
    && apt update \
    && apt install -y wine \
    && rm -rf /var/lib/apt/lists/*

# Install PDAL
RUN apt-get update \
    && apt-get install -y pdal

# Download lastools
RUN wget http://lastools.org/download/lastools.zip && unzip lastools.zip

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
