#!/usr/bin/env python

"""
terra.heightmap.py
This extractor will trigger when a PLY file is uploaded into Clowder. It will create a bmp file to represent heightmap in 2D.
"""

import os
import logging
import subprocess

from pyclowder.utils import CheckMessage
from pyclowder.datasets import get_info, upload_metadata
from pyclowder.files import upload_to_dataset
from terrautils.extractors import TerrarefExtractor, is_latest_file, build_dataset_hierarchy, \
    build_metadata


class heightmap(TerrarefExtractor):
    def __init__(self):
        super(heightmap, self).__init__()

        # parse command line and load default logging configuration
        self.setup(sensor="laser3d_heightmap")

    # Check whether dataset already has metadata
    def check_message(self, connector, host, secret_key, resource, parameters):
        # Check not an bmp file already
        ds_md = get_info(connector, host, secret_key, resource['parent']['id'])
        timestamp = ds_md['name'].split(" - ")[1]
        ply_side = 'west' if resource['name'].find('west') > -1 else 'east'
        out_bmp = self.sensors.get_sensor_path(timestamp, opts=[ply_side])

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
        ply_side = 'west' if resource['name'].find('west') > -1 else 'east'
        out_bmp = self.sensors.create_sensor_path(timestamp, ext='', opts=[ply_side])
        mask_bmp = out_bmp.replace(".bmp", "_mask.bmp")

        logging.info("./main -i %s -o %s" % (input_ply, out_bmp.replace(".bmp", "")))
        subprocess.call(["./main -i %s -o %s" % (input_ply, out_bmp.replace(".bmp", ""))], shell=True)

        files_created = []

        print(out_bmp)
        print(mask_bmp)

        print(os.path.isfile(out_bmp))
        print(os.path.isfile(mask_bmp))

        # the subprocess actually adds to the out_bmp string to create 2 files
        if os.path.isfile(out_bmp):
            self.created += 1
            self.bytes += os.path.getsize(out_bmp)
            # Send bmp output to Clowder source dataset if not already pointed to
            if out_bmp not in resource["local_paths"]:
                print("uploading %s to dataset" % out_bmp)
                fileid = upload_to_dataset(connector, host, secret_key, resource['parent']['id'], out_bmp)
                files_created.append(fileid)

        if os.path.isfile(mask_bmp):
            self.created += 1
            self.bytes += os.path.getsize(mask_bmp)
            # Send bmp output to Clowder source dataset if not already pointed to
            if mask_bmp not in resource["local_paths"]:
                print("uploading %s to dataset" % mask_bmp)
                fileid = upload_to_dataset(connector, host, secret_key, resource['parent']['id'], mask_bmp)
                files_created.append(fileid)

        # Tell Clowder this is completed so subsequent file updates don't daisy-chain
        extmd = build_metadata(host, self.extractor_info, resource['parent']['id'], {
            "files_created": files_created}, 'dataset')
        upload_metadata(connector, host, secret_key, resource['parent']['id'], extmd)

        self.end_message()


if __name__ == "__main__":
    extractor = heightmap()
    extractor.start()
