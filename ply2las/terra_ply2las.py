#!/usr/bin/env python

import os
import logging
import shutil
import subprocess

from pyclowder.utils import CheckMessage
from pyclowder.files import upload_to_dataset
from pyclowder.datasets import upload_metadata
from terrautils.extractors import TerrarefExtractor, is_latest_file, \
    build_dataset_hierarchy, build_metadata


def add_local_arguments(parser):
    # add any additional arguments to parser
    parser.add_argument('--dockerpdal', dest="pdal_docker", type=bool, nargs='?', default=False,
                        help="whether PDAL should be run inside a docker container")

class Ply2LasConverter(TerrarefExtractor):
    def __init__(self):
        super(Ply2LasConverter, self).__init__()

        add_local_arguments(self.parser)

        # parse command line and load default logging configuration
        self.setup(sensor="scanner3DTop_mergedlas")

        # assign other arguments
        self.pdal_docker = self.args.pdal_docker

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
            out_las = self.sensors.get_sensor_path(timestamp, opts=['merged'])
            if os.path.exists(out_las) and not self.overwrite:
                logging.info("output LAS file already exists; skipping %s" % resource['id'])
            else:
                return CheckMessage.download

        return CheckMessage.ignore

    def process_message(self, connector, host, secret_key, resource, parameters):
        self.start_message()
        uploaded_file_ids = []

        east_ply = None
        west_ply = None
        for p in resource['local_paths']:
            if p.endswith(".ply"):
                if p.find("east") > -1:
                    east_ply = p
                elif p.find("west") > -1:
                    west_ply = p

        # Create output in same directory as input, but check name
        timestamp = resource['dataset_info']['name'].split(" - ")[1]
        out_las = self.sensors.get_sensor_path(timestamp, opts=['merged'])
        self.sensors.create_sensor_path(out_las)

        if not os.path.exists(out_las) or self.overwrite:
            if self.args.pdal_docker:
                pdal_base = "docker run -v /home/extractor:/data pdal/pdal:1.5 "
                in_east = east_ply.replace("/home/extractor/sites", "/data/sites")
                in_west = west_ply.replace("/home/extractor/sites", "/data/sites")
                tmp_east_las = "/data/east_temp.las"
                tmp_west_las = "/data/west_temp.las"
                merge_las = "/data/merged.las"
            else:
                pdal_base = ""
                in_east = east_ply
                in_west = west_ply
                tmp_east_las = "east_temp.las"
                tmp_west_las = "west_temp.las"
                merge_las = "/home/extractor/merged.las"

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

            logging.info("merging %s + %s into %s" % (tmp_east_las, tmp_west_las, merge_las))
            subprocess.call([pdal_base+'pdal merge ' + \
                             tmp_east_las+' '+tmp_west_las+' '+merge_las], shell=True)
            if os.path.exists(merge_las):
                shutil.move(merge_las, out_las)
                logging.info("...created %s" % out_las)
                if os.path.isfile(out_las) and out_las not in resource["local_paths"]:
                    target_dsid = build_dataset_hierarchy(connector, host, secret_key, self.clowderspace,
                                                          self.sensors.get_display_name(), timestamp[:4], timestamp[:7],
                                                          timestamp[:10], leaf_ds_name=resource['dataset_info']['name'])

                    # Send LAS output to Clowder source dataset
                    fileid = upload_to_dataset(connector, host, secret_key, target_dsid, out_las)
                    uploaded_file_ids.append(fileid)

            self.created += 1
            self.bytes += os.path.getsize(out_las)

            if os.path.exists(tmp_east_las):
                os.remove(tmp_east_las)
            if os.path.exists(tmp_west_las):
                os.remove(tmp_west_las)

            # Tell Clowder this is completed so subsequent file updates don't daisy-chain
            metadata = build_metadata(host, self.extractor_info['name'], target_dsid, {
                "files_created": [fileid]}, 'dataset')
            upload_metadata(connector, host, secret_key, target_dsid, metadata)

            self.end_message()


if __name__ == "__main__":
    extractor = Ply2LasConverter()
    extractor.start()
