#!/usr/bin/env python

import os
import logging
import numpy as np

from pyclowder.utils import CheckMessage
from pyclowder.files import upload_to_dataset
from pyclowder.datasets import upload_metadata
from terrautils.extractors import TerrarefExtractor, is_latest_file, create_image, \
    build_metadata, calculate_gps_bounds, calculate_centroid, calculate_scan_time, \
    build_dataset_hierarchy, geom_from_metadata
from terrautils.geostreams import create_datapoint_with_dependencies
from terrautils.metadata import get_extractor_metadata

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

            logging.info(out_hist)
            logging.info(out_top)

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
                all_dsmd = full_day_to_histogram.load_json(fname)
                metadata = get_extractor_metadata(all_dsmd, self.extractor_info['name'])
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
        out_hist = self.sensors.create_sensor_path(timestamp, opts=['histogram'], ext='.npy')
        out_top = self.sensors.create_sensor_path(timestamp, opts=['highest'], ext='.npy')

        logging.info("Loading %s & calculating height information" % ply_west)
        metadata = {
            "lemnatec_measurement_metadata": {
                "user_given_metadata": {
                    "experiment title": "Wheat field experiment 2",
                    "experiment responsible": "to be named",
                    "project": "TERRA-REF",
                    "instrument": "gantry at Maricopa phenotyping facility",
                    "location": "Maricopa phenotyping facility",
                    "date of sowing": "2017-04-20",
                    "date of emergence": "2017-04-26",
                    "campaign": "mid season canopy growth",
                    "mission or scan": ""
                },
                "gantry_system_fixed_metadata": {
                    "system manufacturer": "LemnaTec Corp., 4240 Duncan Ave, Saint Louis, MO 63110",
                    "LemnaTec ProjNo": "7100019",
                    "LemnaTecs field scanalzyer no": "2",
                    "project responsible": "ben.niehaus@lemnatec.de",
                    "date of installation": "april 2016",
                    "system scan area n-s [m]": "211",
                    "system scan area e-w [m]": "23.125",
                    "system scan area height [m]": "8",
                    "funded by": "ARPA-E",
                    "trivials": "biggest agriculture robot in the world",
                    "date of handover": "15-12-2016"
                },
                "gantry_system_variable_metadata": {
                    "time": "07/01/2017 00:00:54",
                    "position x [m]": "94.099",
                    "position y [m]": "22.135",
                    "position z [m]": "4.615",
                    "speed x [m/s]": "0",
                    "speed y [m/s]": "0",
                    "speed z [m/s]": "0",
                    "camera box light 1 is on": "False",
                    "camera box light 2 is on": "False",
                    "camera box light 3 is on": "False",
                    "camera box light 4 is on": "False",
                    "Script path on local disk": "C:\\LemnaTec\\StoredScripts\\3D_Scan_Field_SouthStart_033MperS.cs",
                    "Script copy path on FTP server": "ftp://10.160.21.2//gantry_data/LemnaTec/ScriptBackup/3D_Scan_Field_SouthStart_033MperS_470164ba-ad9f-4956-9a51-3e651b3e6c64.cs",
                    "scanIsInPositiveDirection": "False",
                    "scanDistance [m]": "21.8",
                    "scanSpeed [m/s]": "0.33",
                    "scanMode": "Triggered",
                    "sensor setting file path": "c:\\LemnaTec\\StoredSensorSettings\\3dTop.xml"
                },
                "sensor_fixed_metadata": {
                    "sensor manufacturer": "Fraunhofer-Entwicklungszentrum R?ntgentechnik EZRT, Ber?hrungslose Mess- und Pr?fsysteme, www.iis.fraunhofer.de",
                    "sensor product name": "Custom made 3D Scanner",
                    "sensor east serial number": "S215-1030",
                    "sensor west serial number": "S215-1036",
                    "sensor description": "two laser / camera combinations",
                    "sensor purpose": "measure 3D surface topology",
                    "laser emission wavelenght [nm]": "810",
                    "laser emission energy [W]": "2",
                    "location in gantry system": "camera box, facing ground",
                    "scanner east location in camera box x [m]": "2.070",
                    "scanner east location in camera box y [m]": "0.306",
                    "scanner east location in camera box z [m]": "1.135",
                    "scanner west location in camera box x [m]": "2.070",
                    "scanner west location in camera box y [m]": "2.726",
                    "scanner west location in camera box z [m]": "1.135",
                    "field of view y [m]": "0.800",
                    "Calibration available": "true",
                    "output data format": ".ply open file format",
                    "sensor id": "3d camera box"
                },
                "sensor_variable_metadata": {
                    "current setting Exposure [microS]": "70",
                    "current setting Calculate 3D files": "0",
                    "current setting Laser detection threshold": "512",
                    "current setting Scanlines per output file": "100000",
                    "current setting Scan direction (automatically set at runtime)": "0",
                    "current setting Scan distance (automatically set at runtime) [mm]": "21800",
                    "current setting Scan speed (automatically set at runtime) [microMeter/s]": "100000"
                }
            }
        }

        # GEOM DATA
        # gantry_x, gantry_y, gantry_z, cambox_x, cambox_y, cambox_z, fov_x, fov_y = geom_from_metadata(metadata)
        # z_height = gantry_z + cambox_z
        z_height = 4.615 + 1.135
        plydata = PlyData.read(str(ply_west))
        scanDirection = full_day_to_histogram.get_direction(metadata)
        hist, highest = full_day_to_histogram.gen_height_histogram_for_Roman(plydata, scanDirection, 'w', z_height)

        # TODO: Store root collection name in sensors.py?
        target_dsid = build_dataset_hierarchy(connector, host, secret_key, self.clowderspace,
                                              self.sensors.get_display_name(), timestamp[:4], timestamp[:7],
                                              timestamp[:10], leaf_ds_name=resource['dataset_info']['name'])

        if not os.path.exists(out_hist) or self.overwrite:
            np.save(out_hist, hist)
            #create_image(hist, out_hist, scaled=False)
            self.created += 1
            self.bytes += os.path.getsize(out_hist)
            if out_hist not in resource["local_paths"]:
                fileid = upload_to_dataset(connector, host, secret_key, target_dsid, out_hist)
                uploaded_file_ids.append(fileid)

        if not os.path.exists(out_top) or self.overwrite:
            np.save(out_top, highest)
            #create_image(highest, out_top, scaled=False)
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
        metadata = build_metadata(host, self.extractor_info, target_dsid, {
            "files_created": uploaded_file_ids}, 'dataset')
        upload_metadata(connector, host, secret_key, target_dsid, metadata)

        self.end_message()


if __name__ == "__main__":
    extractor = Ply2HeightEstimation()
    extractor.start()
