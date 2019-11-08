#!/usr/bin/env python

import os
import json
import subprocess
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from pyclowder.utils import CheckMessage
from pyclowder.files import upload_to_dataset
from pyclowder.datasets import upload_metadata, download_metadata, remove_metadata
from terrautils.metadata import get_terraref_metadata, get_extractor_metadata, get_season_and_experiment
from terrautils.extractors import TerrarefExtractor, is_latest_file, contains_required_files, \
    build_dataset_hierarchy_crawl, build_metadata, load_json_file, file_exists, check_file_in_dataset
from terrautils.spatial import geom_from_metadata

import leafAngleDistribution as lad


class Ply2LeafAngle(TerrarefExtractor):
    def __init__(self):
        super(Ply2LeafAngle, self).__init__()

        # parse command line and load default logging configuration
        self.setup(sensor="laser3d_leafangle")

    def check_message(self, connector, host, secret_key, resource, parameters):
        if "rulechecked" in parameters and parameters["rulechecked"]:
            return CheckMessage.download

        if not is_latest_file(resource):
            self.log_skip(resource, "not latest file")
            return CheckMessage.ignore

        # Check if we have 2 PLY files, but not an LAS file already
        if not contains_required_files(resource, ['east_0.ply', 'west_0.ply']):
            self.log_skip(resource, "missing required files")
            return CheckMessage.ignore

        # Check metadata to verify we have what we need
        md = download_metadata(connector, host, secret_key, resource['id'])
        if get_terraref_metadata(md):
            if get_extractor_metadata(md, self.extractor_info['name'], self.extractor_info['version']):
                # Make sure outputs properly exist
                timestamp = resource['dataset_info']['name'].split(" - ")[1]
                las = self.sensors.get_sensor_path(timestamp)
                if file_exists(las) and not self.overwrite:
                    self.log_skip(resource, "metadata v%s and outputs already exist" % self.extractor_info['version'])
                    return CheckMessage.ignore
            # Have TERRA-REF metadata, but not any from this extractor
            return CheckMessage.download
        else:
            self.log_skip(resource, "no terraref metadata found")
            return CheckMessage.ignore

    def process_message(self, connector, host, secret_key, resource, parameters):
        self.start_message(resource)

        # Get PLY files and metadata
        west_ply, west_g_png, west_p_png, terra_md_full = None, None, None, None
        for p in resource['local_paths']:
            if p.endswith('_dataset_metadata.json'):
                all_dsmd = load_json_file(p)
                terra_md_full = get_terraref_metadata(all_dsmd, "scanner3DTop")
            elif p.endswith("west_0.ply"):
                west_ply = p
        if west_ply:
            # Must locate PNG files in another directory (not included in dataset)
            seek_dir = os.path.dirname(west_ply).replace("Level_1", "raw_data")
            contents = os.listdir(seek_dir)
            for f in contents:
                if f.endswith("-west_0_g.png"):
                    west_g_png = os.path.join(seek_dir, f)
                elif f.endswith("-west_0_p.png"):
                    west_p_png = os.path.join(seek_dir, f)
        if None in [west_ply, west_g_png, west_p_png, terra_md_full]:
            raise ValueError("could not locate all files & metadata in processing")

        timestamp = resource['dataset_info']['name'].split(" - ")[1]

        # Fetch experiment name from terra metadata
        season_name, experiment_name, updated_experiment = get_season_and_experiment(timestamp, 'scanner3DTop', terra_md_full)
        if None in [season_name, experiment_name]:
            raise ValueError("season and experiment could not be determined")

        # Determine output directory
        self.log_info(resource, "Hierarchy: %s / %s / %s / %s / %s / %s / %s" % (season_name, experiment_name, self.sensors.get_display_name(),
                                                                                 timestamp[:4], timestamp[5:7], timestamp[8:10], timestamp))
        target_dsid = build_dataset_hierarchy_crawl(host, secret_key, self.clowder_user, self.clowder_pass, self.clowderspace,
                                                    season_name, experiment_name, self.sensors.get_display_name(),
                                                    timestamp[:4], timestamp[5:7], timestamp[8:10],
                                                    leaf_ds_name=self.sensors.get_display_name()+' - '+timestamp)
        out_las = self.sensors.create_sensor_path(timestamp)
        out_dir = os.path.dirname(out_las)
        uploaded_file_ids = []

        # Extract position and y offset from metadata
        (gantry_x, gantry_y, gantry_z, cambox_x, cambox_y, cambox_z, fov_x, fov_y) = geom_from_metadata(terra_md_full)
        position = (float(gantry_x) + float(cambox_x), float(gantry_y), float(gantry_z))
        scandirection = int(terra_md_full['sensor_variable_metadata']['scan_direction'])
        y_offset = -25.711 if scandirection == 0 else -3.60

        lad.create_angle_data(out_dir, west_ply, west_g_png, west_p_png, position, y_offset)






        # Attach LemnaTec source metadata to Level_1 product
        self.log_info(resource, "uploading LemnaTec metadata to ds [%s]" % target_dsid)
        remove_metadata(connector, host, secret_key, target_dsid, self.extractor_info['name'])
        terra_md_trim = get_terraref_metadata(all_dsmd)
        if updated_experiment is not None:
            terra_md_trim['experiment_metadata'] = updated_experiment
        terra_md_trim['raw_data_source'] = host + ("" if host.endswith("/") else "/") + "datasets/" + resource['id']
        level1_md = build_metadata(host, self.extractor_info, target_dsid, terra_md_trim, 'dataset')
        upload_metadata(connector, host, secret_key, target_dsid, level1_md)

        if not file_exists(out_las) or self.overwrite:
            # Perform actual processing
            self.log_info(resource, "creating & uploading %s" % out_las)
            self.execute_threaded_conversion([east_ply, west_ply], out_las, terra_md_full)

            # Only upload the newly generated file to Clowder if it isn't already in dataset
            if file_exists(out_las):
                found_in_dest = check_file_in_dataset(connector, host, secret_key, target_dsid, out_las, remove=self.overwrite)
                if not found_in_dest or self.overwrite:
                    fileid = upload_to_dataset(connector, host, secret_key, target_dsid, out_las)
                    uploaded_file_ids.append(host + ("" if host.endswith("/") else "/") + "files/" + fileid)
                self.created += 1
                self.bytes += os.path.getsize(out_las)

        # Tell Clowder this is completed so subsequent file updates don't daisy-chain
        extractor_md = build_metadata(host, self.extractor_info, target_dsid, {
            "files_created": uploaded_file_ids
        }, 'dataset')
        self.log_info(resource, "uploading extractor metadata to raw dataset")
        remove_metadata(connector, host, secret_key, resource['id'], self.extractor_info['name'])
        upload_metadata(connector, host, secret_key, resource['id'], extractor_md)

        self.end_message(resource)

if __name__ == "__main__":
    extractor = Ply2LeafAngle()
    extractor.start()
