'''
Created on Aug 11, 2016

@author: Zongyang
'''
import os
import logging
import full_day_to_histogram
import numpy as np
from config import *
from plyfile import PlyData, PlyElement

import pyclowder.extractors as extractors

def main():
    global extractorName, messageType, rabbitmqExchange, rabbitmqURL, registrationEndpoints

    #set logging
    logging.basicConfig(format='%(levelname)-7s : %(name)s -  %(message)s', level=logging.WARN)
    logging.getLogger('pyclowder.extractors').setLevel(logging.INFO)

    #connect to rabbitmq
    extractors.connect_message_bus(extractorName=extractorName, messageType=messageType, processFileFunction=process_dataset,
                                   checkMessageFunction=check_message, rabbitmqExchange=rabbitmqExchange, rabbitmqURL=rabbitmqURL)

def check_message(parameters):
    # TODO: re-enable once this is merged into Clowder: https://opensource.ncsa.illinois.edu/bitbucket/projects/CATS/repos/clowder/pull-requests/883/overview
    # fetch metadata from dataset to check if we should remove existing entry for this extractor first
    md = extractors.download_dataset_metadata_jsonld(parameters['host'], parameters['secretKey'], parameters['datasetId'], extractorName)
    for m in md:
        if 'agent' in m and 'name' in m['agent']:
            if m['agent']['name'].find(extractorName) > -1:
                print("skipping, already done")
                return False
                #extractors.remove_dataset_metadata_jsonld(parameters['host'], parameters['secretKey'], parameters['datasetId'], extractorName)

    # Check for a east and west file before beginning processing
    found_east = False
    found_west = False
    for f in parameters['filelist']:
        if 'filename' in f and f['filename'].endswith('-east_0.ply'):
            found_east = True
        elif 'filename' in f and f['filename'].endswith('-west_0.ply'):
            found_west = True

    if found_east and found_west:
        return True
    else:
        return False

def process_dataset(parameters):
    metafile, ply_east, ply_west, metadata = None, None, None, None

    # Get left/right files and metadata
    for f in parameters['files']:
        # First check metadata attached to dataset in Clowder for item of interest
        if f.endswith('_dataset_metadata.json'):
            all_dsmd = full_day_to_histogram.load_json(f)
            for curr_dsmd in all_dsmd:
                if 'content' in curr_dsmd and 'lemnatec_measurement_metadata' in curr_dsmd['content']:
                    metafile = f
                    metadata = curr_dsmd['content']
        # Otherwise, check if metadata was uploaded as a .json file
        elif f.endswith('_metadata.json') and f.find('/_metadata.json') == -1 and metafile is None:
            metafile = f
            metadata = full_day_to_histogram.lower_keys(full_day_to_histogram.load_json(metafile))
        elif f.endswith('-east_0.ply'):
            ply_east = f
        elif f.endswith('-west_0.ply'):
            ply_west = f
    if None in [metafile, ply_east, ply_west, metadata]:
        full_day_to_histogram.fail('Could not find all of east/west/metadata.')
        return

    temp_out_dir = metafile.replace(os.path.basename(metafile), "")
    if not os.path.exists(temp_out_dir):
            os.makedirs(temp_out_dir)

    scanDirection = full_day_to_histogram.get_direction(metadata)
    
    print("Loading ply file...")
    plydata = PlyData.read(ply_west)
    
    print("Generating height histogram")
    hist, heightest = full_day_to_histogram.gen_height_histogram(plydata, scanDirection)
    
    
    histPath = os.path.join(temp_out_dir, 'hist.npy')
    np.save(histPath, hist)
    extractors.upload_file_to_dataset(histPath, parameters)
    print("Uploading outputs to dataset: %s" % histPath)
    
    heightestPath = os.path.join(temp_out_dir, 'top.npy')
    np.save(heightestPath, heightest)
    extractors.upload_file_to_dataset(heightestPath, parameters)
    print("Uploading outputs to dataset: %s" % heightestPath)
    
    # Tell Clowder this is completed so subsequent file updates don't daisy-chain
    metadata = {
        "@context": {
            "@vocab": "https://clowder.ncsa.illinois.edu/clowder/assets/docs/api/index.html#!/files/uploadToDataset"
        },
        "dataset_id": parameters["datasetId"],
        "content": {"status": "COMPLETED"},
        "agent": {
            "@type": "cat:extractor",
            "extractor_id": parameters['host'] + "/api/extractors/" + extractorName
        }
    }
    extractors.upload_dataset_metadata_jsonld(mdata=metadata, parameters=parameters)

if __name__ == "__main__":

    main()