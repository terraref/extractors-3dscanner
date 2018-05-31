#!/usr/bin/env python

import os
import logging

from pyclowder.utils import CheckMessage
from pyclowder.files import upload_to_dataset
from pyclowder.datasets import upload_metadata
from terrautils.metadata import get_terraref_metadata
from terrautils.extractors import TerrarefExtractor, is_latest_file, \
    build_dataset_hierarchy, build_metadata, load_json_file

import terraref.laser3d


def add_local_arguments(parser):
    # add any additional arguments to parser
    parser.add_argument('--dockerpdal', dest="pdal_docker", type=bool, nargs='?', default=False,
                        help="whether PDAL should be run inside a docker container")

class Ply2LasConverter(TerrarefExtractor):
    def __init__(self):
        super(Ply2LasConverter, self).__init__()

        add_local_arguments(self.parser)

        # parse command line and load default logging configuration
        self.setup(sensor="laser3d_mergedlas")

        # assign other arguments
        self.pdal_docker = self.args.pdal_docker

    # Check whether dataset already has metadata
    def check_message(self, connector, host, secret_key, resource, parameters):
        if "rulechecked" in parameters and parameters["rulechecked"]:
            return CheckMessage.download

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
            out_las = self.sensors.get_sensor_path(timestamp)
            if os.path.exists(out_las) and not self.overwrite:
                logging.getLogger(__name__).info("output LAS file already exists; skipping %s" % resource['id'])
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
            elif p.endswith('_dataset_metadata.json'):
                all_dsmd = load_json_file(p)
                terra_md = get_terraref_metadata(all_dsmd)

        # Create output in same directory as input, but check name
        timestamp = resource['dataset_info']['name'].split(" - ")[1]
        out_las = self.sensors.create_sensor_path(timestamp)

        if not os.path.exists(out_las) or self.overwrite:
            terraref.laser3d.generate_las_from_ply([east_ply, west_ply], out_las, 'east', terra_md)

            logging.getLogger(__name__).info("...created %s" % out_las)
            if os.path.isfile(out_las) and out_las not in resource["local_paths"]:
                target_dsid = build_dataset_hierarchy(host, secret_key, self.clowder_user, self.clowder_pass, self.clowderspace,
                                                      self.sensors.get_display_name(),
                                                      timestamp[:4], timestamp[5:7],timestamp[8:10],
                                                      leaf_ds_name=self.sensors.get_display_name()+' - '+timestamp)

                # Send LAS output to Clowder source dataset
                fileid = upload_to_dataset(connector, host, secret_key, target_dsid, out_las)
                uploaded_file_ids.append(fileid)

            self.created += 1
            self.bytes += os.path.getsize(out_las)

            # Tell Clowder this is completed so subsequent file updates don't daisy-chain
            metadata = build_metadata(host, self.extractor_info, resource['id'], {
                "files_created": [host + ("" if host.endswith("/") else "/") + "files/" + fileid]}, 'dataset')
            upload_metadata(connector, host, secret_key, resource['id'], metadata)

            # Upload original Lemnatec metadata to new Level_1 dataset
            terra_md['raw_data_source'] = host + ("" if host.endswith("/") else "/") + "datasets/" + resource['id']
            lemna_md = build_metadata(host, self.extractor_info, target_dsid, terra_md, 'dataset')
            upload_metadata(connector, host, secret_key, target_dsid, lemna_md)

            self.end_message()

if __name__ == "__main__":
    extractor = Ply2LasConverter()
    extractor.start()
