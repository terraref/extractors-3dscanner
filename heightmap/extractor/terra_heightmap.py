#!/usr/bin/env python

"""
terra.heightmap.py
This extractor will trigger when a PLY file is uploaded into Clowder. It will create a bmp file to represent heightmap in 2D.
"""

import os
import logging
import subprocess
import tempfile

from pyclowder.extractors import Extractor
from pyclowder.utils import CheckMessage
import pyclowder.files
import pyclowder.datasets



class heightmap(Extractor):
    def __init__(self):
        Extractor.__init__(self)

        # parse command line and load default logging configuration
        self.setup()

        # setup logging for the exctractor
        logging.getLogger('pyclowder').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)
        print "\n 0 \n"

    # Check whether dataset already has metadata
    def check_message(self, connector, host, secret_key, resource, parameters):
        # Check if we have a PLY file, but not an bmp file already

        input_ply = None
        #print resource

        #print("print resource['local_paths']:%s"%(resource["local_paths"][0]))
        for f in resource['files']:
            if f['filename'].endswith(".ply"):
                input_ply=  f['filepath']

        if input_ply:
            out_dir = input_ply.replace(os.path.basename(input_ply), "")
            out_name = resource['name'] + "heightmap.bmp"
            out_bmp = os.path.join(out_dir, out_name)
            if os.path.exists(out_bmp):
                logging.info("output file already exists; skipping %s" % resource['id'])
            else:
                return CheckMessage.download

        return CheckMessage.ignore
                          
    def process_message(self, connector, host, secret_key, resource, parameters):
        input_ply = None
        for p in resource['local_paths']:
            if p.endswith(".ply"):
                input_ply = p
        print input_ply
        print("start processing")
        
        # Create output in same directory as input, but check name
        out_dir = input_ply.replace(os.path.basename(input_ply), "")
        out_name = resource['name'] + "heightmap.bmp"
        out_bmp = os.path.join(out_dir, out_name)


        subprocess.call(['git clone https://github.com/solmazhajmohammadi/heightmap '], shell=True)
        #os.chdir('heightmap')
        subprocess.call(['cp -rT /home/extractor/heightmap .'], shell= True)
        subprocess.call(['chmod 777 main'], shell=True)
        subprocess.call(["./main", "-i", input_ply , "-o", out_bmp])

        if os.path.isfile(out_bmp):
            # Send bmp output to Clowder source dataset
            logging.info("uploading %s to dataset" % out_bmp)
            pyclowder.files.upload_to_dataset(connector, host, secret_key, resource['parent']['id'], out_bmp)
       

if __name__ == "__main__":
    extractor = heightmap()
extractor.start()
