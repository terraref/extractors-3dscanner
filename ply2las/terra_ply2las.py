#!/usr/bin/env python

import datetime
import os
import logging
import shutil
import subprocess

from pyclowder.extractors import Extractor
from pyclowder.utils import CheckMessage
import pyclowder.files
import pyclowder.datasets
import terrautils.extractors


class Ply2LasConverter(Extractor):
    def __init__(self):
        Extractor.__init__(self)

        influx_host = os.getenv("INFLUXDB_HOST", "terra-logging.ncsa.illinois.edu")
        influx_port = os.getenv("INFLUXDB_PORT", 8086)
        influx_db = os.getenv("INFLUXDB_DB", "extractor_db")
        influx_user = os.getenv("INFLUXDB_USER", "terra")
        influx_pass = os.getenv("INFLUXDB_PASSWORD", "")

        # add any additional arguments to parser
        self.parser.add_argument('--output', '-o', dest="output_dir", type=str, nargs='?',
                                 default="/home/extractor/sites/ua-mac/Level_1/scanner3DTop_mergedlas",
                                 help="root directory where timestamp & output directories will be created")
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
        self.output_dir = self.args.output_dir
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
            out_dir = terrautils.extractors.get_output_directory(self.output_dir, resource['name'])
            out_name = terrautils.extractors.get_output_filename(resource['name'], 'las', opts=['merged'])
            out_las = os.path.join(out_dir, out_name)
            if os.path.exists(out_las) and not self.force_overwrite:
                logging.info("output LAS file already exists; skipping %s" % resource['id'])
            else:
                return CheckMessage.download

        return CheckMessage.ignore

    def process_message(self, connector, host, secret_key, resource, parameters):
        starttime = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        created = 0
        bytes = 0

        east_ply = None
        west_ply = None
        for p in resource['local_paths']:
            if p.endswith(".ply"):
                if p.find("east") > -1:
                    east_ply = p
                elif p.find("west") > -1:
                    west_ply = p

        # Create output in same directory as input, but check name
        out_dir = terrautils.extractors.get_output_directory(self.output_dir, resource['name'])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_name = terrautils.extractors.get_output_filename(resource['name'], 'las', opts=['merged'])
        out_las = os.path.join(out_dir, out_name)

        if not os.path.exists(out_las) or self.force_overwrite:
            if self.args.pdal_docker:
                pdal_base = "docker run -v /home/extractor:/data pdal/pdal:1.5 "
                in_east = east_ply.replace("/home/extractor/sites", "/data/sites")
                in_west = west_ply.replace("/home/extractor/sites", "/data/sites")
                tmp_east_las = "/data/east_temp.las"
                tmp_west_las = "/data/west_temp.las"
            else:
                pdal_base = ""
                in_east = east_ply
                in_west = west_ply
                tmp_east_las = "east_temp.las"
                tmp_west_las = "west_temp.las"

            logging.info("converting %s" % east_ply)
            subprocess.call([pdal_base+'pdal translate ' + \
                             '--writers.las.dataformat_id="0" ' + \
                             '--writers.las.scale_x=".000001" ' + \
                             '--writers.las.scale_y=".0001" ' + \
                             '--writers.las.scale_z=".000001" ' + \
                             in_east + " " + tmp_east_las], shell=True)


            logging.info("converting %s" % west_ply)
            subprocess.call([pdal_base+'pdal translate ' + \
                             '--writers.las.dataformat_id="0" ' + \
                             '--writers.las.scale_x=".000001" ' + \
                             '--writers.las.scale_y=".0001" ' + \
                             '--writers.las.scale_z=".000001" ' + \
                             in_west + " " + tmp_west_las], shell=True)

            dock_las = "/data/merged.las"
            logging.info("merging %s + %s into %s" % (tmp_east_las, tmp_west_las, dock_las))
            subprocess.call([pdal_base+'pdal merge ' + \
                             tmp_east_las+' '+tmp_west_las+' '+dock_las], shell=True)
            if os.path.exists("/home/extractor/merged.las"):
                shutil.move("/home/extractor/merged.las", out_las)
                logging.info("...created %s" % out_las)
                if os.path.isfile(out_las) and out_las not in resource["local_paths"]:
                    # Send LAS output to Clowder source dataset
                    fileid = pyclowder.files.upload_to_dataset(connector, host, secret_key, resource['parent']['id'], out_las)

            created += 1
            bytes += os.path.getsize(out_las)

            if os.path.exists(tmp_east_las):
                os.remove(tmp_east_las)
            if os.path.exists(tmp_west_las):
                os.remove(tmp_west_las)

            endtime = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            self.logToInfluxDB(starttime, endtime, created, bytes)

    def logToInfluxDB(self, starttime, endtime, filecount, bytecount):
        # Time of the format "2017-02-10T16:09:57+00:00"
        f_completed_ts = int(parse(endtime).strftime('%s'))*1000000000
        f_duration = f_completed_ts - int(parse(starttime).strftime('%s'))*1000000000

        client = InfluxDBClient(self.influx_host, self.influx_port, self.influx_user, self.influx_pass, self.influx_db)
        client.write_points([{
            "measurement": "file_processed",
            "time": f_completed_ts,
            "fields": {"value": f_duration}
        }], tags={"extractor": self.extractor_info['name'], "type": "duration"})
        client.write_points([{
            "measurement": "file_processed",
            "time": f_completed_ts,
            "fields": {"value": int(filecount)}
        }], tags={"extractor": self.extractor_info['name'], "type": "filecount"})
        client.write_points([{
            "measurement": "file_processed",
            "time": f_completed_ts,
            "fields": {"value": int(bytecount)}
        }], tags={"extractor": self.extractor_info['name'], "type": "bytes"})

if __name__ == "__main__":
    extractor = Ply2LasConverter()
    extractor.start()
