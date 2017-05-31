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


class heightmap(Extractor):
    def __init__(self):
        Extractor.__init__(self)

        influx_host = os.getenv("INFLUXDB_HOST", "terra-logging.ncsa.illinois.edu")
        influx_port = os.getenv("INFLUXDB_PORT", 8086)
        influx_db = os.getenv("INFLUXDB_DB", "extractor_db")
        influx_user = os.getenv("INFLUXDB_USER", "terra")
        influx_pass = os.getenv("INFLUXDB_PASSWORD", "")

        # add any additional arguments to parser
        self.parser.add_argument('--output', '-o', dest="output_dir", type=str, nargs='?',
                                 default="/home/extractor/sites/ua-mac/Level_1/scanner3DTop_heightmap",
                                 help="root directory where timestamp & output directories will be created")
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
        self.output_dir = self.args.output_dir
        self.influx_params = {
            "host": self.args.influx_host,
            "port": self.args.influx_port,
            "db": self.args.influx_db,
            "user": self.args.influx_user,
            "pass": self.args.influx_pass
        }

    # Check whether dataset already has metadata
    def check_message(self, connector, host, secret_key, resource, parameters):
        # Check if we have a PLY file, but not an bmp file already
        input_ply = None
        for f in resource['files']:
            if f['filename'].endswith(".ply"):
                input_ply=  f['filepath']

        if input_ply:
            ds_md = pyclowder.datasets.get_info(connector, host, secret_key, resource['parent']['id'])
            out_dir = terrautils.extractors.get_output_directory(self.output_dir, ds_md['name'])
            out_name = terrautils.extractors.get_output_filename(ds_md['name'], 'bmp', opts=['heightmap'])
            out_bmp = os.path.join(out_dir, out_name)
            if os.path.exists(out_bmp):
                logging.info("output file already exists; skipping %s" % resource['id'])
            else:
                return CheckMessage.download

        return CheckMessage.ignore
                          
    def process_message(self, connector, host, secret_key, resource, parameters):
        starttime = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        created = 0
        bytes = 0

        input_ply = None
        for p in resource['local_paths']:
            if p.endswith(".ply"):
                input_ply = p

        print input_ply
        print("start processing")
        
        # Create output in same directory as input, but check name
        ds_md = pyclowder.datasets.get_info(connector, host, secret_key, resource['parent']['id'])
        out_dir = terrautils.extractors.get_output_directory(self.output_dir, ds_md['name'])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_name = terrautils.extractors.get_output_filename(ds_md['name'], 'bmp', opts=['heightmap'])
        out_bmp = os.path.join(out_dir, out_name)

        subprocess.call(["./main", "-i", input_ply , "-o", out_bmp])

        if os.path.isfile(out_bmp):
            created += 1
            bytes += os.path.getsize(out_bmp)

            # Send bmp output to Clowder source dataset if not already pointed to
            if out_bmp not in resource["local_paths"]:
                logging.info("uploading %s to dataset" % out_bmp)
                pyclowder.files.upload_to_dataset(connector, host, secret_key, resource['parent']['id'], out_bmp)

        endtime = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        self.logToInfluxDB(starttime, endtime, created, bytes)

if __name__ == "__main__":
    extractor = heightmap()
    extractor.start()
