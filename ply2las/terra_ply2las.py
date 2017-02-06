#!/usr/bin/env python

"""
terra.ply2las.py

This extractor will trigger when a PLY file is uploaded into Clowder. It will create an LAS file from the PLY file.
"""

import os
import logging
import subprocess
import tempfile

from pyclowder.extractors import Extractor
from pyclowder.utils import CheckMessage
import pyclowder.files
import pyclowder.datasets


class Ply2LasConverter(Extractor):
    def __init__(self):
        Extractor.__init__(self)

        # parse command line and load default logging configuration
        self.setup()

        # setup logging for the exctractor
        logging.getLogger('pyclowder').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)

    # Check whether dataset already has metadata
    def check_message(self, connector, host, secret_key, resource, parameters):
        # Check if we have 2 PLY files, but not an LAS file already
        east_ply = None
        west_ply = None
        for p in resource['files']:
            if p.endswith(".ply"):
                if p['filename'].find("east") > -1:
                    east_ply = p['filepath']
                elif p['filename'].find("west") > -1:
                    west_ply = p['filepath']

        if east_ply and west_ply:
            out_dir = east_ply.replace(os.path.basename(east_ply), "")
            out_name = resource['name'] + " MergedPointCloud.las"
            out_las = os.path.join(out_dir, out_name)

            if os.path.exists(out_las):
                logging.info("...output file already exists; skipping %s" % resource['id'])
            else:
                return CheckMessage.download

        return CheckMessage.ignore

    def process_message(self, connector, host, secret_key, resource, parameters):
        east_ply = None
        west_ply = None
        for p in resource['local_paths']:
            if p.endswith(".ply"):
                if p.find("east") > -1:
                    east_ply = p
                elif p.find("west") > -1:
                    west_ply = p

        # Create output in same directory as input, but check name
        out_dir = east_ply.replace(os.path.basename(east_ply), "")
        out_name = resource['name'] + " MergedPointCloud.las"
        out_las = os.path.join(out_dir, out_name)

        if east_ply and west_ply:
            tmp_east_las = "east_temp.las"
            logging.info("...converting %s to %s" % (east_ply, tmp_east_las))
            subprocess.call(['pdal translate ' + \
                             '--writers.las.dataformat_id="0" ' + \
                             '--writers.las.scale_x=".000001" ' + \
                             '--writers.las.scale_y=".0001" ' + \
                             '--writers.las.scale_z=".000001" ' + \
                             east_ply + " " + tmp_east_las], shell=True)

            tmp_west_las = "west_temp.las"
            logging.info("...converting %s to %s" % (west_ply, tmp_west_las))
            subprocess.call(['pdal translate ' + \
                             '--writers.las.dataformat_id="0" ' + \
                             '--writers.las.scale_x=".000001" ' + \
                             '--writers.las.scale_y=".0001" ' + \
                             '--writers.las.scale_z=".000001" ' + \
                             west_ply + " " + tmp_west_las], shell=True)

            logging.info("...merging into %s" % out_las)
            subprocess.call(['pdal merge '+tmp_east_las+' '+tmp_west_las+' '+out_las], shell=True)
            if os.path.isfile(out_las):
                # Send LAS output to Clowder source dataset
                logging.info("uploading %s to dataset" % out_las)
                pyclowder.files.upload_to_dataset(connector, host, secret_key, resource['parent']['id'], out_las)

            if os.path.exists(tmp_east_las):
                os.remove(tmp_east_las)
            if os.path.exists(tmp_west_las):
                os.remove(tmp_west_las)

if __name__ == "__main__":
    extractor = Ply2LasConverter()
    extractor.start()
