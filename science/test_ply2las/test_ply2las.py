import unittest
import os
import logging



from terrautils.metadata import get_terraref_metadata, clean_metadata
from terrautils.extractors import load_json_file
from science.ply2las import generate_las_from_pdal, combine_east_west_las, geo_referencing_las, \
    geo_referencing_las_for_eachpoint_in_mac

test_id = '85f9c8c2-fa68-48a6-b63c-375daa438414'
path = os.path.join(os.path.dirname(__file__), test_id)
dire = os.path.join(os.path.dirname(__file__))


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

        generate_las_from_pdal(pdal_base, in_east, tmp_east_las)
        self.assertTrue(os.path.isfile(dire + '/east_temp.las'))

        generate_las_from_pdal(pdal_base, in_west, tmp_west_las)
        self.assertTrue(os.path.isfile(dire + '/west_temp.las'))

        combine_east_west_las(pdal_base, tmp_east_las, tmp_west_las, merge_las)
        self.assertTrue(os.path.isfile(dire + '/merged.las'))

        logging.getLogger(__name__).info("converting LAS coordinates")
        point_cloud_origin = terra_md['sensor_variable_metadata']['point_cloud_origin_m']['east']

        geo_referencing_las(dire + '/merged.las', convert_las, point_cloud_origin)
        geo_referencing_las_for_eachpoint_in_mac(convert_las, convert_pt_las, point_cloud_origin)
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
