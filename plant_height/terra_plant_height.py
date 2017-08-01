#!/usr/bin/env python

import datetime
import os
import logging
import numpy as np

from pyclowder.extractors import Extractor
from pyclowder.utils import CheckMessage
import pyclowder.files
import pyclowder.datasets
import terrautils.extractors
import terrautils.geostreams
import terrautils.sensors
import terrautils.metadata

from plyfile import PlyData, PlyElement
import full_day_to_histogram


class Ply2HeightEstimation(Extractor):
    def __init__(self):
        Extractor.__init__(self)

        influx_host = os.getenv("INFLUXDB_HOST", "terra-logging.ncsa.illinois.edu")
        influx_port = os.getenv("INFLUXDB_PORT", 8086)
        influx_db = os.getenv("INFLUXDB_DB", "extractor_db")
        influx_user = os.getenv("INFLUXDB_USER", "terra")
        influx_pass = os.getenv("INFLUXDB_PASSWORD", "")

        # add any additional arguments to parser
        self.parser.add_argument('--overwrite', dest="force_overwrite", type=bool, nargs='?', default=False,
                                 help="whether to overwrite output file if it already exists in output directory")
        self.parser.add_argument('--dockerpdal', dest="pdal_docker", type=bool, nargs='?', default=False,
                                 help="whether PDAL should be run inside a docker container")
        self.parser.add_argument('--influxHost', dest="influx_host", type=str, nargs='?',
                                 default=influx_host, help="InfluxDB URL for logging")
        self.parser.add_argument('--influxPort', dest="influx_port", type=int, nargs='?',
                                 default=influx_port, help="InfluxDB port")
        self.parser.add_argument('--influxUser', dest="influx_user", type=str, nargs='?',
                                 default=influx_user, help="InfluxDB username")
        self.parser.add_argument('--influxPass', dest="influx_pass", type=str, nargs='?',
                                 default=influx_pass, help="InfluxDB password")
        self.parser.add_argument('--influxDB', dest="influx_db", type=str, nargs='?',
                                 default=influx_db, help="InfluxDB database")

        # parse command line and load default logging configuration
        self.setup()

        # setup logging for the exctractor
        logging.getLogger('pyclowder').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)

        # assign other arguments
        self.force_overwrite = self.args.force_overwrite
        self.pdal_docker = self.args.pdal_docker
        self.influx_params = {
            "host": self.args.influx_host,
            "port": self.args.influx_port,
            "db": self.args.influx_db,
            "user": self.args.influx_user,
            "pass": self.args.influx_pass
        }

    # Check whether dataset already has metadata
    def check_message(self, connector, host, secret_key, resource, parameters):
        if not terrautils.extractors.is_latest_file(resource):
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
            out_hist = terrautils.sensors.get_sensor_path_by_dataset("ua-mac", "Level_1", resource['dataset_info']['name'],
                                                                     "scanner3DTop_plant_height", 'tif', opts=['histogram'])
            out_top = terrautils.sensors.get_sensor_path_by_dataset("ua-mac", "Level_1", resource['dataset_info']['name'],
                                                                     "scanner3DTop_plant_height", 'tif', opts=['highest'])
            if (not self.force_overwrite) and os.path.isfile(out_hist) and os.path.isfile(out_top):
                logging.info("...outputs already exist; skipping %s" % resource['id'])
            else:
                return CheckMessage.download

        return CheckMessage.ignore

    def process_message(self, connector, host, secret_key, resource, parameters):
        starttime = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        created = 0
        bytes = 0
        uploaded_file_ids = []

        # Get left/right files and metadata
        ply_east, ply_west, metadata = None, None, None
        for fname in resource['local_paths']:
            # First check metadata attached to dataset in Clowder for item of interest
            if fname.endswith('_dataset_metadata.json'):
                all_dsmd = full_day_to_histogram.load_json(fname)
                metadata = terrautils.metadata.get_extractor_metadata(all_dsmd)
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
        out_hist = terrautils.sensors.get_sensor_path_by_dataset("ua-mac", "Level_1", resource['dataset_info']['name'],
                                                                 "scanner3DTop_plant_height", 'tif', opts=['histogram'])
        out_top = terrautils.sensors.get_sensor_path_by_dataset("ua-mac", "Level_1", resource['dataset_info']['name'],
                                                                "scanner3DTop_plant_height", 'tif', opts=['highest'])
        out_dir = os.path.dirname(out_hist)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        logging.info("Loading %s & calculating height information" % ply_west)
        plydata = PlyData.read(str(ply_west))
        scanDirection = full_day_to_histogram.get_direction(metadata)
        hist, highest = full_day_to_histogram.gen_height_histogram(plydata, scanDirection)

        if not os.path.exists(out_hist) or self.force_overwrite:
            terrautils.extractors.create_image(hist, out_hist, scaled=False)
            created += 1
            bytes += os.path.getsize(out_hist)
            if out_hist not in resource["local_paths"]:
                fileid = pyclowder.files.upload_to_dataset(connector, host, secret_key, resource['id'], out_hist)
                uploaded_file_ids.append(fileid)

        if not os.path.exists(out_top) or self.force_overwrite:
            terrautils.extractors.create_image(highest, out_top, scaled=False)
            created += 1
            bytes += os.path.getsize(out_top)
            if out_top not in resource["local_paths"]:
                fileid = pyclowder.files.upload_to_dataset(connector, host, secret_key, resource['id'], out_top)
                uploaded_file_ids.append(fileid)

        # Prepare and submit datapoint
        left_bounds = terrautils.extractors.calculate_gps_bounds(metadata)[0]
        sensor_latlon = terrautils.extractors.calculate_centroid(left_bounds)
        logging.info("sensor lat/lon: %s" % str(sensor_latlon))

        fileIdList = []
        for f in resource['files']:
            fileIdList.append(f['id'])
        # Format time properly, adding UTC if missing from Danforth timestamp
        ctime = terrautils.extractors.calculate_scan_time(metadata)
        time_obj = time.strptime(ctime, "%m/%d/%Y %H:%M:%S")
        time_fmt = time.strftime('%Y-%m-%dT%H:%M:%S', time_obj)
        if len(time_fmt) == 19:
            time_fmt += "-06:00"

        dpmetadata = {
            "max_height": np.max(highest),
            "source": host+"datasets/"+resource['id'],
            "file_ids": ",".join(fileIdList)
        }
        terrautils.geostreams.create_datapoint_with_dependencies(connector, host, secret_key,
                                                                 "Plant Height", sensor_latlon,
                                                                 time_fmt, time_fmt, dpmetadata)

        # Tell Clowder this is completed so subsequent file updates don't daisy-chain
        metadata = terrautils.extractors.build_metadata(host, self.extractor_info['name'], resource['id'], {
            "files_created": uploaded_file_ids}, 'dataset')
        pyclowder.datasets.upload_metadata(connector, host, secret_key, resource['id'], metadata)

        endtime = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        terrautils.extractors.log_to_influxdb(self.extractor_info['name'], self.influx_params,
                                              starttime, endtime, created, bytes)


if __name__ == "__main__":
    extractor = Ply2HeightEstimation()
    extractor.start()
