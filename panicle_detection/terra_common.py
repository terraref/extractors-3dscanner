'''
Created on Sep 6, 2016

@author: Zongyang Li
'''
import json, sys
from math import cos, pi

"reference coordinates of ranges from TERRA_sorghum_field_book"
# comes from https://docs.google.com/spreadsheets/d/1eQSeVMPfrWS9Li4XlJf3qs2F8txmddbwZhjOfMGAvt8/edit#gid=1765508846
_x_range = [(0, 0, 0),
            (2.3, 3.8, 5.3),
            (6.3, 7.8, 9.3),
            (10.26, 11.76, 13.26),
            (14.22, 15.72, 17.22),
            (18.12, 19.62, 21.12),
            (22.21, 23.71, 25.21),
            (26.26, 27.76, 29.26),
            (30.16, 31.66, 33.16),
            (33.97, 35.47, 36.97),
            (38.14, 39.64, 41.14),
            (42.12, 43.62, 45.12),
            (46.24, 47.74, 49.24),
            (50.21, 51.71, 53.21),
            (54.16, 55.66, 57.16),
            (57.98, 59.48, 60.98),
            (61.86, 63.36, 64.86), 
            (66.03, 67.53, 69.03),
            (70.03, 71.53, 73.03),
            (73.89, 75.39, 76.89),
            (77.85, 79.35, 80.85),
            (81.62, 83.12, 84.62),
            (85.86, 87.36, 88.86),
            (89.68, 91.18, 92.68),
            (93.86, 95.36, 96.86),
            (97.64, 99.14, 100.64),
            (101.72, 103.22, 104.72),
            (105.94, 107.44, 108.94),
            (109.74, 111.24, 112.74),
            (113.7, 115.2, 116.7),
            (117.7, 119.2, 120.7),
            (121.86, 123.36, 124.86),
            (125.81, 127.31, 128.81),
            (129.73, 131.23, 132.73),
            (133.76, 135.26, 136.76),
            (137.6, 139.1, 140.6), 
            (141.64, 143.14, 144.64),
            (145.56, 147.06, 148.56),
            (149.43, 150.93, 152.43),
            (153.39, 154.89, 156.39),
            (157.34, 158.84, 160.34),
            (161.39, 162.89, 164.39),
            (165.45, 166.95, 168.45),
            (169.45, 170.95, 172.45),
            (173.41, 174.91, 176.41),
            (177.55, 179.05, 180.55),
            (181.53, 183.03, 184.53),
            (185.38, 186.88, 188.38),
            (189.38, 190.88, 192.38),
            (193.26, 194.76, 196.26),
            (197.23, 198.73, 200.23),
            (201.36, 202.86, 204.36),
            (205.34, 206.84, 208.34)]
            

_y_row = [24.334,23.569,22.805,22.04,21.275,20.511,19.746,18.982,18.217,17.453,16.688,15.923,15.159,14.394,13.63,
          12.865,12.101,11.336,10.571,9.807,9.042,8.278,7.513,6.749,5.984,5.219,4.455,3.69,2.926,2.161,1.397,0.632, 0.0]

_x_range_s2 =  [(209.857,    213.821),
                (205.874,    209.838),
                (201.890,    205.854),
                (197.906,    201.870),
                (193.923,    197.887),
                (189.939,    193.903),
                (185.955,    189.919),
                (181.972,    185.936),
                (177.988,    181.952),
                (174.005,    177.969),
                (170.021,    173.985),
                (166.037,    170.001),
                (162.054,    166.018),
                (158.070,    162.034),
                (154.086,    158.050),
                (150.103,    154.067),
                (146.119,    150.083),
                (142.136,    146.100),
                (138.152,    142.116),
                (134.168,    138.132),
                (130.185,    134.149),
                (126.201,    130.165),
                (122.217,    126.181),
                (118.234,    122.198),
                (114.250,    118.214),
                (110.266,    114.230),
                (106.283,    110.247),
                (102.299,    106.263),
                (98.316,    102.280),
                (94.332,    98.296),
                (90.348,    94.312),
                (86.365,    90.329),
                (82.381,    86.345),
                (78.397,    82.361),
                (74.414,    78.378),
                (70.430,    74.394),
                (66.446,    70.410),
                (62.463,    66.427),
                (58.479,    62.443),
                (54.496,    58.460),
                (50.512,    54.476),
                (46.528,    50.492),
                (42.545,    46.509),
                (38.561,    42.525),
                (34.577,    38.541),
                (30.594,    34.558),
                (26.610,    30.574),
                (22.627,    26.591),
                (18.643,    22.607),
                (14.659,    18.623),
                (10.676,    14.640),
                (6.692 ,   10.656),
                (2.708 ,   6.672),
                (-1.275,    2.689)]

_x_range_s4 =  [(209.556,    213.541),
            (205.571,    209.556),
            (201.586,    205.571),
            (197.602,    201.586),
            (193.617,    197.602),
            (189.632,    193.617),
            (185.647,    189.632),
            (181.663,    185.647),
            (177.678,    181.663),
            (173.693,    177.678),
            (169.708,    173.693),
            (165.724,    169.708),
            (161.739,    165.724),
            (157.754,    161.739),
            (153.769,    157.754),
            (149.785,    153.769),
            (145.800,    149.785),
            (141.815,    145.800),
            (137.830,    141.815),
            (133.846,    137.830),
            (129.861,    133.846),
            (125.876,    129.861),
            (121.891,    125.876),
            (117.907,    121.891),
            (113.922,    117.907),
            (109.937,    113.922),
            (105.952,    109.937),
            (101.967,    105.952),
            (97.983,    101.967),
            (93.998,    97.983),
            (90.013,    93.998),
            (86.028,    90.013),
            (82.044,    86.028),
            (78.059,    82.044),
            (74.074,    78.059),
            (70.089,    74.074),
            (66.105,    70.089),
            (62.120,    66.105),
            (58.135,    62.120),
            (54.150,    58.135),
            (50.166,    54.150),
            (46.181,    50.166),
            (42.196,    46.181),
            (38.211,    42.196),
            (34.227,    38.211),
            (30.242,    34.227),
            (26.257,    30.242),
            (22.272,    26.257),
            (18.288,    22.272),
            (14.303,    18.288),
            (10.318,    14.303),
            (6.333,    10.318),
            (2.348,    6.333)]

_y_row_s2 =[(23.952,    24.717),
            (23.187,    23.952),
            (22.423,    23.187),
            (21.658,    22.423),
            (20.893,    21.658),
            (20.123,    20.893),
            (19.364,    20.123),
            (18.600,    19.364),
            (17.835,    18.600),
            (17.071,    17.835),
            (16.306,    17.071),
            (15.541,    16.306),
            (14.777,    15.541),
            (14.012,    14.777),
            (13.248,    14.012),
            (12.483,    13.248),
            (11.719,    12.483),
            (10.954,    11.719),
            (10.189,    10.954),
            (9.425,     10.189),
            (8.660,     9.425),
            (7.896,     8.660),
            (7.131,     7.896),
            (6.367,     7.131),
            (5.602,     6.367),
            (4.837,     5.602),
            (4.073,     4.837),
            (3.308,     4.073),
            (2.544,     3.308),
            (1.779,     2.544),
            (1.015,     1.779),
            (0.250,     1.015)]

_y_row_s4 =[(24.064,    24.828),
            (23.300,    24.064),
            (22.536,    23.300),
            (21.772,    22.536),
            (21.007,    21.772),
            (20.243,    21.007),
            (19.479,    20.243),
            (18.715,    19.479),
            (17.950,    18.715),
            (17.186,    17.950),
            (16.422,    17.186),
            (15.658,    16.422),
            (14.894,    15.658),
            (14.129,    14.894),
            (13.365,    14.129),
            (12.601,    13.365),
            (11.837,    12.601),
            (11.073,    11.837),
            (10.308,    11.073),
            (9.544,    10.308),
            (8.780,    9.544),
            (8.016,    8.780),
            (7.252,    8.016),
            (6.487,    7.252),
            (5.723,    6.487),
            (4.959,    5.723),
            (4.195,    4.959),
            (3.431,    4.195),
            (2.666,    3.431),
            (1.902,    2.666),
            (1.138,    1.902),
            (0.374,    1.138)]

"(latitude, longitude) of SE corner (positions are + in NW direction)"
ZERO_ZERO = (33.0745,-111.97475)

class CoordinateConverter(object):
    """
        This class implements coordinate conversions
        what coordinate system do we have in terra?
        LatLon, field_position, field_partition, plot_number, pixels
        
        LatLon: latitude/longitude, EPSG:4326, ZERO_ZERO = (33.0745,-111.97475)  SE corner (positions are + in NW direction)
        field_position: field position in meters which gantry system is using, ZERO_ZERO = (3.8, 0.0)
        field_partition: field are divided into 54 rows and 16 columns, partition(1, 1) is in SW corner (+ in NE direction)
        plot_number: a number that created by field_partition, details are in the file "Sorghum TERRA plot plan 2016 4.21.16.xlsx"
        pixels: (x,y,z) coordinate in different sensor data, 2-D or 3-D, need a parameter of 'field of view' or other parameter.
    """
    
    def __init__(self):
        self.fov = 0
        self.pixSize = 0.9853
        
    def fieldPosition_to_Latlon(self, x, y):
        "Converts field position to latlon"
        r = 6378137 # earth's radius
        
        x_min = y
        y_min = x

        lat_min_offset = y_min/r* 180/pi
        lng_min_offset = x_min/(r * cos(pi * ZERO_ZERO[0]/180)) * 180/pi

        lat = ZERO_ZERO[0] - lat_min_offset
        lng = ZERO_ZERO[1] - lng_min_offset
        
        return lat, lng
    
    def fieldPosition_to_fieldPartition(self, x, y):
        
        plot_row = 0
        plot_col = 0
        count = 0
        
        for (xmin, xmid, xmax) in _x_range:
            count = count + 1
            if (x > xmin) and (x < xmax):
                plot_row = count
                break
        
        count = 0
        for ymax in _y_row:
            count = count + 1
            if(y > ymax):
                plot_col = count/2
                break
        
        return plot_row, plot_col
    
    
    def plotNum_to_fieldPartition(self, plotNum):
        "Converts plot number to field partition"
        cols = 16
        col = plotNum % cols
        if col == 0:
            plot_row = plotNum / cols
            if (plot_row % 2 == 0):
                plot_col = 1
            else:
                plot_col = 16
                
            return plot_row, plot_col
        
        
        plot_row = plotNum/cols +1
        plot_col = col
        if (plot_row % 2 == 0):
            plot_col = cols - col + 1
        
        return plot_row, plot_col
    
    def fieldPartition_to_fieldPosition(self, plot_row, plot_col):
        "Converts field partition to field position"
        if plot_row < 1 or plot_row > 54 or plot_col < 1 or plot_col > 16:
            return [0, 0], [0, 0]
        
        x_position = [_x_range[plot_row-1][0], _x_range[plot_row-1][2]]
        y_position = [_y_row[2*(plot_col)], _y_row[2*plot_col-1]]
        
        return x_position, y_position
    
    def pixel_to_plotNum(self, x, y, position, fov, scan_d, x_width, y_width):
        "Converts pixel to plot number, given pixel x, y, field position, sensor field of view, scan distance, image width and height"
        x_param = float(fov) / float(x_width)
        y_param = float(scan_d) / float(y_width)
        
        x_shift = x * x_param
        y_shift = y * y_param
        
        x_real = position[0] - float(fov)/2 + x_shift
        y_real = position[1] + y_shift
            
        plot_row, plot_col = self.fieldPosition_to_fieldPartition(x_real, y_real)
            
        plotNum = self.fieldPartition_to_plotNum(plot_row, plot_col)
        
        return plotNum
    
    def fieldPartition_to_plotNum(self, plot_row, plot_col):
        "Converts field partition to plot number"
        if plot_row == 0:
            return 0
        
        if plot_row % 2 == 0:
            plot_col = 17 - plot_col
            
        plotNum = (plot_row-1)*16 + plot_col
    
        return plotNum
    
    def fieldPartition_to_plotNum_32(self, plot_row, plot_col):
        "Converts field partition to plot number"
        if plot_row == 0:
            return 0
        
        if plot_row % 2 == 0:
            plot_col = 33 - plot_col
            
        plotNum = (plot_row-1)*32 + plot_col
    
        return plotNum


def load_json(meta_path):
    try:
        with open(meta_path, 'r') as fin:
            return json.load(fin)
    except Exception as ex:
        fail('Corrupt metadata file, ' + str(ex))
        
def fail(reason):
    print >> sys.stderr, reason
    
    
def lower_keys(in_dict):
    if type(in_dict) is dict:
        out_dict = {}
        for key, item in in_dict.items():
            out_dict[key.lower()] = lower_keys(item)
        return out_dict
    elif type(in_dict) is list:
        return [lower_keys(obj) for obj in in_dict]
    else:
        return in_dict