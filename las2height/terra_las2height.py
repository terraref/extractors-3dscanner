#!/usr/bin/env python

import os
from laspy.file import File
import numpy as np

from pyclowder.utils import CheckMessage
from pyclowder.files import upload_to_dataset
from pyclowder.datasets import upload_metadata, download_metadata, remove_metadata
from terrautils.extractors import TerrarefExtractor, is_latest_file, create_image, \
    build_metadata, calculate_gps_bounds, calculate_centroid, calculate_scan_time, \
    build_dataset_hierarchy, geom_from_metadata, contains_required_files, file_exists, \
    load_json_file, check_file_in_dataset
from terrautils.metadata import get_terraref_metadata

#TODO: from terraref.laser3d import las_to_height


class LAS2HeightEstimation(TerrarefExtractor):
    def __init__(self):
        super(LAS2HeightEstimation, self).__init__()

        # parse command line and load default logging configuration
        self.setup(sensor="laser3d_canopyheight")

    # Check whether dataset already has metadata
    def check_message(self, connector, host, secret_key, resource, parameters):
        if not is_latest_file(resource):
            return CheckMessage.ignore

        # Check if we have 2 PLY files, but not an LAS file already
        if not contains_required_files(resource, ['.las']):
            self.log_skip(resource, "missing required files")
            return CheckMessage.ignore

        # Check metadata to verify we have what we need
        md = download_metadata(connector, host, secret_key, resource['id'])
        if get_terraref_metadata(md):
            # Have TERRA-REF metadata, but not any from this extractor
            return CheckMessage.download
        else:
            self.log_skip(resource, "no terraref metadata found")
            return CheckMessage.ignore

    def process_message(self, connector, host, secret_key, resource, parameters):
        self.start_message()
        uploaded_file_ids = []

        # Load metadata from dataset
        las_file, terra_md_full = None, None
        for fname in resource['local_paths']:
            if fname.endswith('.las'):
                las_file = fname
            elif fname.endswith('_dataset_metadata.json'):
                all_dsmd = load_json_file(fname)
                terra_md_full = get_terraref_metadata(all_dsmd)

        # Determine script name
        target_scan = "unknown_scan"
        if 'gantry_variable_metadata' in terra_md_full:
            if 'script_name' in terra_md_full['gantry_variable_metadata']:
                target_scan = terra_md_full['gantry_variable_metadata']['script_name']
                if 'script_hash' in terra_md_full['gantry_variable_metadata']:
                    target_scan += ' '+terra_md_full['gantry_variable_metadata']['script_hash']

        # Determine output locations
        plotname = resource['dataset_info']['name'].split(" - ")[1]
        date = resource['dataset_info']['name'].split(" - ")[2]
        out_hist = self.sensors.create_sensor_path(date, plot=plotname, filename=target_scan+"_histogram.csv")

        hist = las_to_height_distribution(las_file, out_hist)
        maximum = np.max(hist)
        self.log_info(resource, "maximum height found: %s" % maximum)

        found_in_dest = check_file_in_dataset(connector, host, secret_key, resource['id'], out_hist, remove=self.overwrite)
        if not found_in_dest or self.overwrite:
            fileid = upload_to_dataset(connector, host, secret_key, resource['id'], out_hist)
            uploaded_file_ids.append(host + ("" if host.endswith("/") else "/") + "files/" + fileid)
        self.created += 1
        self.bytes += os.path.getsize(out_hist)

        # TODO: Submit highest value histogram to BETYdb as a trait

        # Tell Clowder this is completed so subsequent file updates don't daisy-chain
        extractor_md = build_metadata(host, self.extractor_info, resource['id'], {
            "files_created": uploaded_file_ids
        }, 'dataset')
        self.log_info(resource, "uploading extractor metadata to Level_1 dataset")
        remove_metadata(connector, host, secret_key, resource['id'], self.extractor_info['name'])
        upload_metadata(connector, host, secret_key, resource['id'], extractor_md)

        self.end_message()

def las_to_height_distribution(in_file, out_file):
    HIST_BIN_NUM = 500
    zRange = [0, 5000] # depth scope of laser scanner, unit in mm
    zOffset = 10  # bin width of histogram, 10 mm for each bin
    scaleParam = 1000 # min unit in las might be 0.001 mm
    height_hist = np.zeros((zRange[1]-zRange[0])/zOffset)

    las_handle = File(in_file)
    zData = las_handle.Z

    if (zData.size) == 0:
        return height_hist

    with open(out_file, 'w') as out:
        out.write("bin,range,pts\n")

        for i in range(0, HIST_BIN_NUM):
            zmin = i * zOffset * scaleParam
            zmax = (i+1) * zOffset * scaleParam
            if i == 0:
                zIndex = np.where(zData<zmax)
            elif i == HIST_BIN_NUM-1:
                zIndex = np.where(zData>zmin)
            else:
                zIndex = np.where((zData>zmin) & (zData<zmax))
            num = len(zIndex[0])
            height_hist[i] = num

            out.write("%s,%s,%s\n" % (i, "%s-%s" % (zmin, zmax), num))

    return height_hist

if __name__ == "__main__":
    extractor = LAS2HeightEstimation()
    extractor.start()
