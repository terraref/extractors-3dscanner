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
    # Check if LAS equivalent already exists - need
    in_ply = parameters['inputfile']
    out_las = in_ply.replace('.ply', '.las').replace('.PLY', '.LAS')

    if not os.path.isfile(out_las):
        return True
    else:
        print("%s already exists; skipping" % out_las)
        return False
    """

def process_file(parameters):
    in_ply = parameters['inputfile']

    # Create output in same directory as input, but check name
    out_dir = in_ply.replace(os.path.basename(in_ply), "")
    out_name = parameters['filename'].replace('.ply', '.las').replace('.PLY', '.LAS')
    out_las = os.path.join(out_dir, out_name)

    if in_ply:
        print("converting %s to %s" % (in_ply, out_las))

        if not os.path.isfile(out_las):
            # Execute processing on target file
            subprocess.call(['pdal translate ' + \
                             '--writers.las.dataformat_id="0" ' + \
                             '--writers.las.scale_x=".000001" ' + \
                             '--writers.las.scale_y=".0001" ' + \
                             '--writers.las.scale_z=".000001" ' + \
                             in_ply + " " + out_las], shell=True)

            if os.path.isfile(out_las):
                # Send LAS output to Clowder source dataset
                print("uploading %s to dataset" % out_las)
                extractors.upload_file_to_dataset(out_las, parameters)
        else:
            print("...%s already exists; skipping" % out_las)

if __name__ == "__main__":
    main()
