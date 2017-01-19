#!/usr/bin/env python

"""
terra.ply2las.py

This extractor will trigger when a PLY file is uploaded into Clowder. It will create an LAS file from the PLY file.
"""

import os
import logging
import subprocess

from pyclowder.extractors import Extractor
from pyclowder.utils import CheckMessage
import pyclowder.files
import pyclowder.datasets


class Ply2LasConverter(Extractor):
    def __init__(self):
        Extractor.__init__(self)

        # add any additional arguments to parser
        # self.parser.add_argument('--max', '-m', type=int, nargs='?', default=-1,
        #                          help='maximum number (default=-1)')
        self.parser.add_argument('--output', '-o', dest="output_dir", type=str, nargs='?',
                                 default="/home/extractor/sites/ua-mac/Level_1/netcdf",
                                 help="root directory where timestamp & output directories will be created")

        # parse command line and load default logging configuration
        self.setup()

        # setup logging for the exctractor
        logging.getLogger('pyclowder').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)

        # assign other arguments
        self.output_dir = self.args.output_dir

    # Check whether dataset already has metadata
    def check_message(self, connector, host, secret_key, resource, parameters):
        # For now if the dataset already has metadata from this extractor, don't recreate
        md = pyclowder.datasets.download_metadata(connector, host, secret_key,
                                                  parameters['datasetId'], self.extractor_info['name'])
        if len(md) > 0:
            for m in md:
                if 'agent' in m and 'name' in m['agent']:
                    if m['agent']['name'].find(self.extractor_info['name']) > -1:
                        logging.info("skipping file %s, already processed" % resource['id'])
                        return CheckMessage.ignore

        return CheckMessage.download

    def process_message(self, connector, host, secret_key, resource, parameters):
        in_ply = resource['local_paths'][0]

        # Create output in same directory as input, but check name
        out_dir = in_ply.replace(os.path.basename(in_ply), "")
        out_name = resource['name'].replace('.ply', '.las').replace('.PLY', '.LAS')
        out_las = os.path.join(out_dir, out_name)

        if in_ply:
            logging.info("...converting %s to %s" % (in_ply, out_las))

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
                    logging.info("uploading %s to dataset" % out_las)
                    pyclowder.files.upload_to_dataset(connector, host, secret_key, resource['parent']['id'], out_las)
            else:
                logging.info("...%s already exists; skipping" % out_las)

if __name__ == "__main__":
    extractor = Ply2LasConverter()
    extractor.start()
