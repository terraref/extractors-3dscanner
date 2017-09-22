#!/usr/bin/env python

"""
terra.heightmap.py
This extractor will trigger when a PLY file is uploaded into Clowder. It will create a bmp file to represent heightmap in 2D.
"""

import os
import logging
import subprocess
import numpy
from PIL import Image

from pyclowder.utils import CheckMessage
from pyclowder.datasets import get_info, upload_metadata, download_metadata
from pyclowder.files import upload_to_dataset
from terrautils.extractors import TerrarefExtractor, is_latest_file, build_dataset_hierarchy, \
    build_metadata, load_json_file
from terrautils.metadata import get_terraref_metadata, clean_metadata, get_sensor_fixed_metadata
from terrautils.formats import create_geotiff
from terrautils.spatial import geojson_to_tuples


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
        out_tif = self.sensors.get_sensor_path(timestamp, opts=[ply_side])

        if os.path.exists(out_tif):
            logging.info("output file already exists; skipping %s" % resource['id'])
            return CheckMessage.ignore

        return CheckMessage.download
                          
    def process_message(self, connector, host, secret_key, resource, parameters):
        self.start_message()

        input_ply = resource['local_paths'][0]
        
        # Create output in same directory as input, but check name
        ds_md = get_info(connector, host, secret_key, resource['parent']['id'])
        terra_md = get_terraref_metadata(download_metadata(connector, host, secret_key,
                                                           resource['parent']['id']), 'scanner3DTop')
        if terra_md == {}:
            # Load & clean metadata.json file in equivalent raw_data directory
            ply_dir = os.path.dirname(resource['local_paths'][0])
            md_dir = ply_dir.replace("Level_1", "raw_data")
            for mdf in os.listdir(md_dir):
                if mdf.endswith("metadata.json"):
                    terra_md = clean_metadata(load_json_file(os.path.join(md_dir,mdf)), "scanner3DTop")
                    # Go ahead and add it to the dataset
                    cleaned_md = {
                        "@context": ["https://clowder.ncsa.illinois.edu/contexts/metadata.jsonld",
                                     {"@vocab": "https://terraref.ncsa.illinois.edu/metadata/uamac#"}],
                        "content": terra_md,
                        "agent": {
                            "@type": "cat:user",
                            "user_id": "https://terraref.ncsa.illinois.edu/clowder/api/users/57adcb81c0a7465986583df1"
                        }
                    }
                    upload_metadata(connector, host, secret_key, resource['parent']['id'], cleaned_md)
                    terra_md['sensor_fixed_metadata'] = get_sensor_fixed_metadata("ua-mac", "scanner3DTop")
            if terra_md == {}:
                logging.error("no metadata found")
                return False


        timestamp = ds_md['name'].split(" - ")[1]
        ply_side = 'west' if resource['name'].find('west') > -1 else 'east'

        if ply_side not in terra_md['spatial_metadata']:
            logging.error("incompatible metadata format")
            return False

        gps_bounds = geojson_to_tuples(terra_md['spatial_metadata'][ply_side]['bounding_box'])
        out_tif = self.sensors.create_sensor_path(timestamp, ext='', opts=[ply_side])
        mask_tif = out_tif.replace(".tif", "_mask.tif")
        out_bmp = out_tif.replace(".tif", ".bmp")
        mask_bmp = out_tif.replace(".tif", "_mask.bmp")
        files_created = []

        # Create BMPs first
        logging.info("./main -i %s -o %s" % (input_ply, out_bmp.replace(".bmp", "")))
        subprocess.call(["./main -i %s -o %s" % (input_ply, out_bmp.replace(".bmp", ""))], shell=True)

        # Then convert BMP images to GeoTIFFs (flipping negative direction scans 180 degress)
        with Image.open(out_bmp) as bmp:
            px_array = numpy.array(bmp)
            px_array = numpy.rot90(px_array, 3)
            create_geotiff(px_array, gps_bounds, out_tif)
        os.remove(out_bmp)
        with Image.open(mask_bmp) as bmp:
            px_array = numpy.array(bmp)
            px_array = numpy.rot90(px_array, 3)
            create_geotiff(px_array, gps_bounds, mask_tif)
        os.remove(mask_bmp)

        # Upload all 2 outputs
        """
        if os.path.isfile(out_bmp):
            self.created += 1
            self.bytes += os.path.getsize(out_bmp)
            # Send bmp output to Clowder source dataset if not already pointed to
            if out_bmp not in resource["local_paths"]:
                fileid = upload_to_dataset(connector, host, secret_key, resource['parent']['id'], out_bmp)
                files_created.append(fileid)
        if os.path.isfile(mask_bmp):
            self.created += 1
            self.bytes += os.path.getsize(mask_bmp)
            # Send bmp output to Clowder source dataset if not already pointed to
            if mask_bmp not in resource["local_paths"]:
                fileid = upload_to_dataset(connector, host, secret_key, resource['parent']['id'], mask_bmp)
                files_created.append(fileid)
        """
        if os.path.isfile(out_tif):
            self.created += 1
            self.bytes += os.path.getsize(out_tif)
            # Send bmp output to Clowder source dataset if not already pointed to
            if out_tif not in resource["local_paths"]:
                fileid = upload_to_dataset(connector, host, secret_key, resource['parent']['id'], out_tif)
                files_created.append(fileid)
        if os.path.isfile(mask_tif):
            self.created += 1
            self.bytes += os.path.getsize(mask_tif)
            # Send bmp output to Clowder source dataset if not already pointed to
            if mask_tif not in resource["local_paths"]:
                fileid = upload_to_dataset(connector, host, secret_key, resource['parent']['id'], mask_tif)
                files_created.append(fileid)

        # Tell Clowder this is completed so subsequent file updates don't daisy-chain
        extmd = build_metadata(host, self.extractor_info, resource['parent']['id'], {
            "files_created": files_created}, 'dataset')
        upload_metadata(connector, host, secret_key, resource['parent']['id'], extmd)

        self.end_message()


if __name__ == "__main__":
    extractor = heightmap()
    extractor.start()
