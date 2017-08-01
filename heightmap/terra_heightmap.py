#!/usr/bin/env python

"""
terra.heightmap.py
This extractor will trigger when a PLY file is uploaded into Clowder. It will create a bmp file to represent heightmap in 2D.
"""

import datetime
import os
import logging
import subprocess

from pyclowder.extractors import Extractor
from pyclowder.utils import CheckMessage
import pyclowder.files
import pyclowder.datasets
import terrautils.extractors
import terrautils.sensors


class heightmap(Extractor):
    def __init__(self):
        Extractor.__init__(self)

        influx_host = os.getenv("INFLUXDB_HOST", "terra-logging.ncsa.illinois.edu")
        influx_port = os.getenv("INFLUXDB_PORT", 8086)
        influx_db = os.getenv("INFLUXDB_DB", "extractor_db")
        influx_user = os.getenv("INFLUXDB_USER", "terra")
        influx_pass = os.getenv("INFLUXDB_PASSWORD", "")

        # add any additional arguments to parser
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
        self.influx_params = {
            "host": self.args.influx_host,
            "port": self.args.influx_port,
            "db": self.args.influx_db,
            "user": self.args.influx_user,
            "pass": self.args.influx_pass
        }

    # Check whether dataset already has metadata
    def check_message(self, connector, host, secret_key, resource, parameters):
        # Check not an bmp file already
        ds_md = pyclowder.datasets.get_info(connector, host, secret_key, resource['parent']['id'])
        out_bmp = terrautils.sensors.get_sensor_path_by_dataset("ua-mac", "Level_1", ds_md['name'],
                                                                "scanner3DTop_heightmap", '', opts=['heightmap'])

        if os.path.exists(out_bmp):
            logging.info("output file already exists; skipping %s" % resource['id'])
            return CheckMessage.ignore
        else:
            return CheckMessage.download
                          
    def process_message(self, connector, host, secret_key, resource, parameters):
        starttime = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        created = 0
        bytes = 0

        input_ply = resource['local_paths'][0]
        
        # Create output in same directory as input, but check name
        ds_md = pyclowder.datasets.get_info(connector, host, secret_key, resource['parent']['id'])
        out_bmp = terrautils.sensors.get_sensor_path_by_dataset("ua-mac", "Level_1", ds_md['name'],
                                                                "scanner3DTop_heightmap", '', opts=['heightmap'])
        out_dir = os.path.dirname(out_bmp)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        logging.info("./main -i %s -o %s" % (input_ply, out_bmp))
        subprocess.call(["./main -i %s -o %s" % (input_ply, out_bmp)], shell=True)

        # the subprocess actually adds to the out_bmp string to create 2 files
        main_bmp = out_bmp+".bmp"
        if os.path.isfile(main_bmp):
            created += 1
            bytes += os.path.getsize(main_bmp)
            # Send bmp output to Clowder source dataset if not already pointed to
            if main_bmp not in resource["local_paths"]:
                logging.info("uploading %s to dataset" % main_bmp)
                pyclowder.files.upload_to_dataset(connector, host, secret_key, resource['parent']['id'], main_bmp)

        mask_bmp = out_bmp+"_mask.bmp"
        if os.path.isfile(mask_bmp):
            created += 1
            bytes += os.path.getsize(mask_bmp)
            # Send bmp output to Clowder source dataset if not already pointed to
            if mask_bmp not in resource["local_paths"]:
                logging.info("uploading %s to dataset" % mask_bmp)
                pyclowder.files.upload_to_dataset(connector, host, secret_key, resource['parent']['id'], mask_bmp)

        endtime = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        terrautils.extractors.log_to_influxdb(self.extractor_info['name'], self.influx_params,
                                              starttime, endtime, created, bytes)

if __name__ == "__main__":
    extractor = heightmap()
    extractor.start()
