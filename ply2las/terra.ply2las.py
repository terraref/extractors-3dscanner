#!/usr/bin/env python

"""
terra.ply2las.py

This extractor will trigger when a PLY file is uploaded into Clowder. It will create an LAS file from the PLY file.
"""

import os
import imp
import logging
import requests
import subprocess

from config import *
import pyclowder.extractors as extractors


def main():
    global extractorName, messageType, rabbitmqExchange, rabbitmqURL, registrationEndpoints, mountedPaths

    #set logging
    logging.basicConfig(format='%(levelname)-7s : %(name)s -  %(message)s', level=logging.WARN)
    logging.getLogger('pyclowder.extractors').setLevel(logging.INFO)
    logger = logging.getLogger('extractor')
    logger.setLevel(logging.DEBUG)

    # setup
    extractors.setup(extractorName=extractorName,
                     messageType=messageType,
                     rabbitmqURL=rabbitmqURL,
                     rabbitmqExchange=rabbitmqExchange,
                     mountedPaths=mountedPaths)

    # register extractor info
    extractors.register_extractor(registrationEndpoints)

    #connect to rabbitmq
    extractors.connect_message_bus(extractorName=extractorName,
                                   messageType=messageType,
                                   processFileFunction=process_file,
                                   checkMessageFunction=check_message,
                                   rabbitmqExchange=rabbitmqExchange,
                                   rabbitmqURL=rabbitmqURL)

# ----------------------------------------------------------------------
def check_message(parameters):
    return True

    """
    # Check if LAS equivalent already exists
    in_ply = parameters['inputfile']
    out_las = in_ply.replace('.ply', '.las').replace('.PLY', '.LAS')

    if not os.path.isfile(out_las):
        return True
    else:
        print("%s already exists; skipping" % out_las)
        return False
    """

def process_file(parameters):
    global outputDir

    in_ply = parameters['inputfile']
    out_las = in_ply.replace('.ply', '.las').replace('.PLY', '.LAS')

    if in_ply:
        """ From Solmaz
        --- PREP
        module load pdal <ROGER>
            or - apt-get install pdal <LINUX>
            or - brew install pdal <OSX>

        --- CONVERSION
        pdal translate --writers.las.dataformat_id="0" --writers.las.scale_x=".0000001" --writers.las.scale_y=".0001" --writers.las.scale_z=".000001" input.ply output.las

        --- MERGING (DISABLED)
        Following command will merge the pointclouds:
            pdal merge east.las west.las merged.las
        But, my suggestion is to keep the LAS files separately for East and West camera, due to the time difference between the scans.

        --- VERSION OPTIONS
        0 == no color or time stored
        1 == time is stored
        2 == color is stored
        3 == color and time are stored
        6 == time is stored (version 1.4+ only)
        7 == time and color are stored (version 1.4+ only)
        8 == time, color and near infrared are stored (version 1.4+ only)

        Choose the scale based on the precision that you want to save the file.
        The default LAS version is "2", you can change it to "1" with adding:
            --writers.las.minor_version="1"
        """

        print("converting %s to %s" % (in_ply, out_las))

        if not os.path.isfile(out_las):
            # Execute processing on target file
            subprocess.call(['pdal', 'translate',
                             '--writers.las.dataformat_id', '0',
                             '--writers.las.scale_x', ".0000001",
                             '--writers.las.scale_y', ".0001",
                             '--writers.las.scale_z', ".000001",
                             in_ply, out_las])

            if os.path.isfile(out_las):
                # Send LAS output to Clowder source dataset
                extractors.upload_file_to_dataset(out_las, parameters)
        else:
            print("...%s already exists; skipping" % out_las)

if __name__ == "__main__":
    main()
