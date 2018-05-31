import unittest
import os
import logging
import subprocess


from terrautils.metadata import get_terraref_metadata, clean_metadata
from terrautils.extractors import load_json_file
from terra_ply2las import Ply2LasConverter

test_id = '85f9c8c2-fa68-48a6-b63c-375daa438414'
path = os.path.join(os.path.dirname(__file__), 'test_ply2las', test_id)
dire = os.path.join(os.path.dirname(__file__), 'test_ply2las')


class TestPly2las(unittest.TestCase):
    def test_ply2las(self):
        all_dsmd = load_json_file(path + '_metadata.json')
        cleanmetadata = clean_metadata(all_dsmd, "scanner3DTop")
        terra_md = get_terraref_metadata(cleanmetadata, 'scanner3DTop')

        east_ply = path + '__Top-heading-east_0.ply'
        west_ply = path + '__Top-heading-west_0.ply'
        in_east = '/data/' + test_id + '__Top-heading-east_0.ply'
        in_west = '/data/' + test_id + '__Top-heading-west_0.ply'

        pdal_base = "docker run -v %s:/data pdal/pdal:1.5 " % dire
        tmp_east_las = "/data/east_temp.las"
        tmp_west_las = "/data/west_temp.las"
        merge_las = "/data/merged.las"
        convert_las = dire+"/converted.las"
        convert_pt_las = dire+"/converted_pts.las"

        logging.getLogger(__name__).info("converting %s" % east_ply)
        subprocess.call([pdal_base+'pdal translate ' +
                         '--writers.las.dataformat_id="0" ' +
                         '--writers.las.scale_x=".000001" ' +
                         '--writers.las.scale_y=".0001" ' +
                         '--writers.las.scale_z=".000001" ' +
                         in_east + " " + tmp_east_las], shell=True)

        self.assertTrue(os.path.isfile(dire + '/east_temp.las'))

        logging.getLogger(__name__).info("converting %s" % west_ply)
        subprocess.call([pdal_base+'pdal translate ' +
                         '--writers.las.dataformat_id="0" ' +
                         '--writers.las.scale_x=".000001" ' +
                         '--writers.las.scale_y=".0001" ' +
                         '--writers.las.scale_z=".000001" ' +
                         in_west + " " + tmp_west_las], shell=True)

        self.assertTrue(os.path.isfile(dire + '/west_temp.las'))

        logging.getLogger(__name__).info("merging %s + %s into %s" % (tmp_east_las, tmp_west_las, merge_las))
        subprocess.call([pdal_base+'pdal merge ' +
                         tmp_east_las+' '+tmp_west_las+' '+merge_las], shell=True)
        self.assertTrue(os.path.isfile(dire + '/merged.las'))

        logging.getLogger(__name__).info("converting LAS coordinates")
        point_cloud_origin = terra_md['sensor_variable_metadata']['point_cloud_origin_m']['east']

        geo_info = Ply2LasConverter()
        geo_info.geo_referencing_las(dire + '/merged.las', convert_las, point_cloud_origin)
        geo_info.geo_referencing_las_for_eachpoint_in_mac(convert_las, convert_pt_las, point_cloud_origin)
        self.assertTrue(os.path.isfile(convert_las))
        self.assertTrue(os.path.isfile(convert_pt_las))

        os.remove(dire + '/east_temp.las')
        os.remove(dire + '/west_temp.las')
        os.remove(dire + '/merged.las')
        os.remove(convert_las)
        os.remove(convert_pt_las)

        return


if __name__ == '__main__':
    unittest.main()
