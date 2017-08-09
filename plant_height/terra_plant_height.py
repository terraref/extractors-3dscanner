#!/usr/bin/env python

import datetime
import os
import logging
import numpy as np


from pyclowder.utils import CheckMessage

import terrautils.extractors
import terrautils.geostreams
import terrautils.sensors
from terrautils.metadata import get_extractor_metadata

from pyclowder.files import upload_to_dataset
from pyclowder.datasets import upload_metadata
from terrautils.extractors import TerrarefExtractor, is_latest_file, create_image, \
    build_metadata, calculate_gps_bounds, calculate_centroid, calculate_scan_time
from terrautils.geostreams import create_datapoint_with_dependencies, build_dataset_hierarchy

from plyfile import PlyData, PlyElement
import full_day_to_histogram


class Ply2HeightEstimation(TerrarefExtractor):
    def __init__(self):
        super(Ply2HeightEstimation, self).__init__()

        # parse command line and load default logging configuration
        self.setup(sensor="scanner3DTop_plant_height")

    # Check whether dataset already has metadata
    def check_message(self, connector, host, secret_key, resource, parameters):
        if not is_latest_file(resource):
            return CheckMessage.ignore

        # Check if we have 2 PLY files, but not an LAS file already
        east_ply = None
        west_ply = None
        for p in resource['files']:
            if p['filename'].endswith(".ply"):
                if p['filename'].find("east") > -1:
                    east_ply = p['filepath']
                elif p['filename'].find("west") > -1:
                    west_ply = p['filepath']

        if east_ply and west_ply:
            timestamp = resource['dataset_info']['name'].split(" - ")[1]
            out_hist = self.sensors.get_sensor_path(timestamp, opts=['histogram'], ext='.tif')
            out_top = self.sensors.get_sensor_path(timestamp, opts=['highest'], ext='.tif')
            if (not self.force_overwrite) and os.path.isfile(out_hist) and os.path.isfile(out_top):
                logging.info("...outputs already exist; skipping %s" % resource['id'])
            else:
                return CheckMessage.download

        return CheckMessage.ignore

    def process_message(self, connector, host, secret_key, resource, parameters):
        self.start_message()
        uploaded_file_ids = []

        # Get left/right files and metadata
        ply_east, ply_west, metadata = None, None, None
        for fname in resource['local_paths']:
            # First check metadata attached to dataset in Clowder for item of interest
            if fname.endswith('_dataset_metadata.json'):
                all_dsmd = full_day_to_histogram.load_json(fname)
                metadata = get_extractor_metadata(all_dsmd)
            # Otherwise, check if metadata was uploaded as a .json file
            elif fname.endswith('_metadata.json') and fname.find('/_metadata.json') == -1 and metadata is None:
                metadata = full_day_to_histogram.lower_keys(full_day_to_histogram.load_json(fname))
            elif fname.endswith('-east_0.ply'):
                ply_east = fname
            elif fname.endswith('-west_0.ply'):
                ply_west = fname
        if None in [ply_east, ply_west, metadata]:
            logging.error('could not find all 3 of east/west/metadata')
            return

        # Determine output locations
        timestamp = resource['dataset_info']['name'].split(" - ")[1]
        out_hist = self.sensors.get_sensor_path(timestamp, opts=['histogram'], ext='.tif')
        out_top = self.sensors.get_sensor_path(timestamp, opts=['highest'], ext='.tif')
        self.sensors.create_sensor_path(out_hist)

        # TODO: Store root collection name in sensors.py?
        target_dsid = build_dataset_hierarchy(connector, host, secret_key, self.clowderspace,
                                              "scanner3DTop plant height", timestamp[:4], timestamp[:7],
                                              timestamp[:10], leaf_ds_name=resource['dataset_info']['name'])

        logging.info("Loading %s & calculating height information" % ply_west)
        plydata = PlyData.read(str(ply_west))
        scanDirection = full_day_to_histogram.get_direction(metadata)
        hist, highest = full_day_to_histogram.gen_height_histogram(plydata, scanDirection)

        if not os.path.exists(out_hist) or self.force_overwrite:
            create_image(hist, out_hist, scaled=False)
            self.created += 1
            self.bytes += os.path.getsize(out_hist)
            if out_hist not in resource["local_paths"]:
                fileid = upload_to_dataset(connector, host, secret_key, target_dsid, out_hist)
                uploaded_file_ids.append(fileid)

        if not os.path.exists(out_top) or self.force_overwrite:
            create_image(highest, out_top, scaled=False)
            self.created += 1
            self.bytes += os.path.getsize(out_top)
            if out_top not in resource["local_paths"]:
                fileid = upload_to_dataset(connector, host, secret_key, target_dsid, out_top)
                uploaded_file_ids.append(fileid)

        # Prepare and submit datapoint
        left_bounds = calculate_gps_bounds(metadata)[0]
        sensor_latlon = calculate_centroid(left_bounds)
        logging.info("sensor lat/lon: %s" % str(sensor_latlon))

        fileIdList = []
        for f in resource['files']:
            fileIdList.append(f['id'])
        # Format time properly, adding UTC if missing from Danforth timestamp
        ctime = calculate_scan_time(metadata)
        time_obj = time.strptime(ctime, "%m/%d/%Y %H:%M:%S")
        time_fmt = time.strftime('%Y-%m-%dT%H:%M:%S', time_obj)
        if len(time_fmt) == 19:
            time_fmt += "-06:00"

        dpmetadata = {
            "max_height": np.max(highest),
            "source": host+"datasets/"+resource['id'],
            "file_ids": ",".join(fileIdList)
        }
        create_datapoint_with_dependencies(connector, host, secret_key,
                                           "Plant Height", sensor_latlon,
                                           time_fmt, time_fmt, dpmetadata)

        # Tell Clowder this is completed so subsequent file updates don't daisy-chain
        metadata = build_metadata(host, self.extractor_info['name'], target_dsid, {
            "files_created": uploaded_file_ids}, 'dataset')
        upload_metadata(connector, host, secret_key, target_dsid, metadata)

        self.end_message()


if __name__ == "__main__":
    extractor = Ply2HeightEstimation()
    extractor.start()
