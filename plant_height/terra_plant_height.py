#!/usr/bin/env python

import os
import logging
import json
import numpy as np

from pyclowder.utils import CheckMessage
from pyclowder.files import upload_to_dataset
from pyclowder.datasets import upload_metadata
from terrautils.extractors import TerrarefExtractor, is_latest_file,build_metadata
from terrautils.spatial import calculate_gps_bounds, calculate_centroid, geom_from_metadata
from terrautils.geostreams import create_datapoint_with_dependencies
from terrautils.metadata import get_terraref_metadata, calculate_scan_time


from plyfile import PlyData
from scanner_3d.plant_height import load_json, lower_keys, get_direction, gen_height_histogram_for_Roman


class Ply2HeightEstimation(TerrarefExtractor):
    def __init__(self):
        super(Ply2HeightEstimation, self).__init__()

        # parse command line and load default logging configuration
        self.setup(sensor="laser3d_plant_height")

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
            out_hist = self.sensors.get_sensor_path(timestamp, opts=['histogram'], ext='.json')
            out_top = self.sensors.get_sensor_path(timestamp, opts=['highest'], ext='.json')

            if (not self.overwrite) and os.path.isfile(out_hist) and os.path.isfile(out_top):
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
                all_dsmd = load_json(fname)
                metadata = get_terraref_metadata(all_dsmd, 'scanner3DTop')
            # Otherwise, check if metadata was uploaded as a .json file
            elif fname.endswith('_metadata.json') and fname.find('/_metadata.json') == -1 and metadata is None:
                metadata = lower_keys(load_json(fname))
            elif fname.endswith('-east_0.ply'):
                ply_east = fname
            elif fname.endswith('-west_0.ply'):
                ply_west = fname
        if None in [ply_east, ply_west, metadata]:
            logging.error('could not find all 3 of east/west/metadata')
            return

        # Determine output locations
        timestamp = resource['dataset_info']['name'].split(" - ")[1]
        out_hist = self.sensors.create_sensor_path(timestamp, opts=['histogram'], ext='.json')
        out_top = self.sensors.create_sensor_path(timestamp, opts=['highest'], ext='.json')

        logging.info("Loading %s & calculating height information" % ply_west)
        gantry_x, gantry_y, gantry_z, cambox_x, cambox_y, cambox_z, fov_x, fov_y = geom_from_metadata(metadata, side='west')
        z_height = float(gantry_z) + float(cambox_z)
        plydata = PlyData.read(str(ply_west))
        scanDirection = get_direction(metadata)

        bounds = calculate_gps_bounds(metadata, 'laser3d_plant_height')
        sensor_latlon = calculate_centroid(bounds)
        logging.info("sensor lat/lon: %s" % str(sensor_latlon))

        hist, highest = gen_height_histogram_for_Roman(plydata, scanDirection, 'w', z_height)
        # Convert numpy arrays to JSON
        highest_json = highest.reshape(1,32).tolist()[0]
        hist_json = hist.tolist()

        if not os.path.exists(out_hist) or self.overwrite:
            #np.save(out_hist, hist)
            with open(out_hist, 'w') as o:
                json.dump(hist_json, o, indent=4)
            self.created += 1
            self.bytes += os.path.getsize(out_hist)
            if out_hist not in resource["local_paths"]:
                fileid = upload_to_dataset(connector, host, secret_key, resource['id'], out_hist)
                uploaded_file_ids.append(fileid)

        if not os.path.exists(out_top) or self.overwrite:
            #np.save(out_top, highest)
            with open(out_top, 'w') as o:
                json.dump(highest_json, o, indent=4)
            self.created += 1
            self.bytes += os.path.getsize(out_top)
            if out_top not in resource["local_paths"]:
                fileid = upload_to_dataset(connector, host, secret_key, resource['id'], out_top)
                uploaded_file_ids.append(fileid)

        # TODO: Submit highest value histogram to BETYdb as a trait


        # Prepare and submit datapoint
        fileIdList = []
        for f in resource['files']:
            fileIdList.append(f['id'])
        # Format time properly, adding UTC if missing from Danforth timestamp
        ctime = calculate_scan_time(metadata)
        dpmetadata = {
            "max_height": np.max(highest),
            "source": host+"datasets/"+resource['id'],
            "file_ids": ",".join(fileIdList)
        }
        create_datapoint_with_dependencies(connector, host, secret_key,
                                           self.sensors.get_display_name(), sensor_latlon,
                                           ctime, ctime, dpmetadata)

        # Tell Clowder this is completed so subsequent file updates don't daisy-chain
        extmd = build_metadata(host, self.extractor_info, resource['id'], {
            "files_created": uploaded_file_ids,
            "max_height": np.max(highest)}, 'dataset')
        upload_metadata(connector, host, secret_key, resource['id'], extmd)

        self.end_message()


if __name__ == "__main__":
    extractor = Ply2HeightEstimation()
    extractor.start()
