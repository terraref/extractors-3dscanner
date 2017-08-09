#!/usr/bin/env python

"""
terra.heightmap.py
This extractor will trigger when a PLY file is uploaded into Clowder. It will create a bmp file to represent heightmap in 2D.
"""

import os
import logging
import subprocess

from pyclowder.utils import CheckMessage
from pyclowder.datasets import get_info
from pyclowder.files import upload_to_dataset
from terrautils.extractors import TerrarefExtractor, is_latest_file, build_dataset_hierarchy


class heightmap(TerrarefExtractor):
    def __init__(self):
        super(heightmap, self).__init__()

        # parse command line and load default logging configuration
        self.setup(sensor="scanner3DTop_heightmap")

    # Check whether dataset already has metadata
    def check_message(self, connector, host, secret_key, resource, parameters):
        # Check not an bmp file already
        ds_md = get_info(connector, host, secret_key, resource['parent']['id'])
        timestamp = ds_md['name'].split(" - ")[1]
        out_bmp = self.sensors.get_sensor_path(timestamp, opts=['heightmap'])

        if os.path.exists(out_bmp):
            logging.info("output file already exists; skipping %s" % resource['id'])
            return CheckMessage.ignore
        else:
            return CheckMessage.download
                          
    def process_message(self, connector, host, secret_key, resource, parameters):
        self.start_message()

        input_ply = resource['local_paths'][0]
        
        # Create output in same directory as input, but check name
        ds_md = get_info(connector, host, secret_key, resource['parent']['id'])
        timestamp = ds_md['name'].split(" - ")[1]
        out_bmp = self.sensors.get_sensor_path(timestamp, opts=['heightmap'])
        self.sensors.create_sensor_path(out_bmp)

        logging.info("./main -i %s -o %s" % (input_ply, out_bmp))
        subprocess.call(["./main -i %s -o %s" % (input_ply, out_bmp)], shell=True)

        # TODO: Store root collection name in sensors.py?
        timestamp = ds_md['name'].split(" - ")[1]
        target_dsid = build_dataset_hierarchy(connector, host, secret_key, self.clowderspace,
                                              "scanner3DTop heightmap", timestamp[:4], timestamp[:7],
                                              timestamp[:10], leaf_ds_name=resource['dataset_info']['name'])

        # the subprocess actually adds to the out_bmp string to create 2 files
        if os.path.isfile(out_bmp):
            self.created += 1
            self.bytes += os.path.getsize(out_bmp)
            # Send bmp output to Clowder source dataset if not already pointed to
            if out_bmp not in resource["local_paths"]:
                logging.info("uploading %s to dataset" % out_bmp)
                upload_to_dataset(connector, host, secret_key, target_dsid, out_bmp)

        mask_bmp = out_bmp.replace(".bmp", "_mask.bmp")
        if os.path.isfile(mask_bmp):
            self.created += 1
            self.bytes += os.path.getsize(mask_bmp)
            # Send bmp output to Clowder source dataset if not already pointed to
            if mask_bmp not in resource["local_paths"]:
                logging.info("uploading %s to dataset" % mask_bmp)
                upload_to_dataset(connector, host, secret_key, target_dsid, mask_bmp)

        self.end_message()


if __name__ == "__main__":
    extractor = heightmap()
    extractor.start()
