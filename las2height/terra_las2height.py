#!/usr/bin/env python

import os

from pyclowder.utils import CheckMessage
from pyclowder.files import submit_extraction
from pyclowder.datasets import upload_metadata, download_metadata, remove_metadata
from terrautils.extractors import TerrarefExtractor, is_latest_file, upload_to_dataset, \
    build_metadata, contains_required_files, file_exists, load_json_file, check_file_in_dataset
from terrautils.betydb import add_arguments

from terraref.laser3d import las_to_height



# TODO: Keep these in terrautils.bety instead
def get_traits_table():
    # Compiled traits table
    fields = ('local_datetime', 'canopy_cover', 'access_level', 'species', 'site',
              'citation_author', 'citation_year', 'citation_title', 'method')
    traits = {'local_datetime' : '',
              'canopy_height' : [],
              'access_level': '2',
              'species': 'Sorghum bicolor',
              'site': [],
              'citation_author': '"Zongyang, Li"',
              'citation_year': '2016',
              'citation_title': 'Maricopa Field Station Data and Metadata',
              'method': 'Canopy Height Estimation from Field Scanner Laser 3D scans'}

    return (fields, traits)

# TODO: Keep these in terrautils.bety instead
def generate_traits_list(traits):
    # compose the summary traits
    trait_list = [  traits['local_datetime'],
                    traits['canopy_cover'],
                    traits['access_level'],
                    traits['species'],
                    traits['site'],
                    traits['citation_author'],
                    traits['citation_year'],
                    traits['citation_title'],
                    traits['method']
                    ]

    return trait_list


def add_local_arguments(parser):
    # add any additional arguments to parser
    add_arguments(parser)

class LAS2HeightEstimation(TerrarefExtractor):
    def __init__(self):
        super(LAS2HeightEstimation, self).__init__()

        add_local_arguments(self.parser)

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
        out_csv = out_hist.replace("_histogram.csv", "_canopyheight_bety.csv")
        uploaded_file_ids = []

        (hist, maximum) = las_to_height(las_file, out_hist)
        self.log_info(resource, "maximum height found: %s" % maximum)

        found_in_dest = check_file_in_dataset(connector, host, secret_key, resource['id'], out_hist, remove=self.overwrite)
        if not found_in_dest or self.overwrite:
            fileid = upload_to_dataset(connector, host, self.clowder_user, self.clowder_pass, resource['id'], out_hist)
            uploaded_file_ids.append(host + ("" if host.endswith("/") else "/") + "files/" + fileid)
        self.created += 1
        self.bytes += os.path.getsize(out_hist)

        self.log_info(resource, "Writing BETY CSV to %s" % out_csv)
        csv_file = open(out_csv, 'w')
        (fields, traits) = get_traits_table()
        csv_file.write(','.join(map(str, fields)) + '\n')
        traits['canopy_cover'] = str(maximum)
        traits['site'] = plotname
        traits['local_datetime'] = date+"T12:00:00"
        trait_list = generate_traits_list(traits)
        csv_file.write(','.join(map(str, trait_list)) + '\n')
        csv_file.close()

        # Upload this CSV to Clowder
        fileid = upload_to_dataset(connector, host, self.clowder_user, self.clowder_pass, resource['id'], out_csv)

        # Tell Clowder this is completed so subsequent file updates don't daisy-chain
        extractor_md = build_metadata(host, self.extractor_info, resource['id'], {
            "files_created": uploaded_file_ids,
            "max_height_cm": "%s" % maximum
        }, 'dataset')
        self.log_info(resource, "uploading extractor metadata to Level_1 dataset")
        remove_metadata(connector, host, secret_key, resource['id'], self.extractor_info['name'])
        upload_metadata(connector, host, secret_key, resource['id'], extractor_md)

        # Trigger separate extractors
        self.log_info(resource, "triggering BETY extractor on %s" % fileid)
        submit_extraction(connector, host, secret_key, fileid, "terra.betydb")

        self.end_message(resource)

if __name__ == "__main__":
    extractor = LAS2HeightEstimation()
    extractor.start()
