#!/usr/bin/env python

import os
from laspy.file import File
import numpy as np

from pyclowder.utils import CheckMessage
from pyclowder.files import upload_to_dataset
from pyclowder.datasets import upload_metadata, download_metadata, remove_metadata
from terrautils.extractors import TerrarefExtractor, is_latest_file, \
    build_metadata, contains_required_files, file_exists, load_json_file, check_file_in_dataset
from terrautils.metadata import get_terraref_metadata

#TODO: from terraref.laser3d import las_to_height


class LAS2HeightEstimation(TerrarefExtractor):
    def __init__(self):
        super(LAS2HeightEstimation, self).__init__()

        # parse command line and load default logging configuration
        self.setup(sensor="laser3d_canopyheight")

    # Check whether dataset already has metadata
    def check_message(self, connector, host, secret_key, resource, parameters):
        if "rulechecked" in parameters and parameters["rulechecked"]:
            return CheckMessage.download

        if not is_latest_file(resource):
            self.log_skip(resource, "not latest file")
            return CheckMessage.ignore

        # Check if we have 2 PLY files, but not an LAS file already
        if not contains_required_files(resource, ['_merged.las']):
            self.log_skip(resource, "missing required files")
            return CheckMessage.ignore

        return CheckMessage.download

    def process_message(self, connector, host, secret_key, resource, parameters):
        self.start_message(resource)

        # Load metadata from dataset
        las_file, terra_md_full = None, None
        for fname in resource['local_paths']:
            if fname.endswith('.las'):
                las_file = fname

        target_scan = os.path.basename(las_file).replace("_merged.las", "")

        # Determine output locations
        plotname = resource['dataset_info']['name'].split(" - ")[1]
        date = resource['dataset_info']['name'].split(" - ")[2]
        out_hist = self.sensors.create_sensor_path(date, plot=plotname, filename=target_scan+"_histogram.csv")
        uploaded_file_ids = []

        (hist, maximum) = las_to_height_distribution(las_file, out_hist)
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
            "files_created": uploaded_file_ids,
            "max_height": "%s" % maximum
        }, 'dataset')
        self.log_info(resource, "uploading extractor metadata to Level_1 dataset")
        remove_metadata(connector, host, secret_key, resource['id'], self.extractor_info['name'])
        upload_metadata(connector, host, secret_key, resource['id'], extractor_md)

        self.end_message(resource)

def las_to_height_distribution(in_file, out_file):
    HIST_BIN_NUM = 500
    zRange = [0, 5000]  # depth scope of laser scanner, unit in mm
    zOffset = 10        # bin width of histogram, 10 mm for each bin
    scaleParam = 1      # min unit in las might be 0.001 mm
    height_hist = np.zeros((zRange[1]-zRange[0])/zOffset)

    las_handle = File(in_file)
    zData = las_handle.Z

    if (zData.size) == 0:
        return height_hist, 0

    max_height = (np.max(zData)/100.0)

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

            out.write("%s,%s,%s\n" % (i, "%s-%s" % (zmin/100.0, zmax/100.0), num))

    return height_hist, max_height

if __name__ == "__main__":
    extractor = LAS2HeightEstimation()
    extractor.start()
