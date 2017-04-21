'''
Created on Aug 29, 2016

@author: Zongyang Li
'''
import os,sys,json,shutil,terra_common
import numpy as np
import requests
from glob import glob
import matplotlib
from matplotlib import pyplot as plt
from plyfile import PlyData, PlyElement

convt = terra_common.CoordinateConverter()

F_SLOPE = 0.661
F_OFFSET = 28.2

def main():
    
    '''
    in_dir = '/Users/Desktop/heightDistribution/heightDistribution/'
    out_dir = '/Users/Desktop/heightDistribution/WEB_GUI Height_Distribution/'
    
    #gen_plot_Xth_percentile(in_dir, out_dir)
    gen_plot_heatmap(in_dir, out_dir)
    #copy_3d_vis(in_dir, out_dir)
    
    '''
    in_dir = ''
    out_dir = ''
    
    insert_height_traits_into_betydb(in_dir, out_dir)
    
    
    return

def copy_3d_vis(in_dir, out_dir):
    
    list_dirs = os.walk(in_dir)
    
    for root, sub_dir, files in list_dirs:
        for d in sub_dir:
            full_path = os.path.join(in_dir, d)
            if not os.path.isdir(full_path):
                continue
            
            scan_date = d
            one_day_copy(full_path, scan_date, out_dir)
    
    
    return

def one_day_copy(in_dir, scan_date, out_dir):
    
    list_dirs = os.walk(in_dir)
    
    for root, dirs, files in list_dirs:
        for d in dirs:
            full_path = os.path.join(in_dir, d)
            if not os.path.isdir(full_path):
                continue
    
            meta_copy_process(full_path, scan_date, out_dir)
    
    return

def meta_copy_process(in_dir, scan_date, out_dir):
    
    metafile = find_files(in_dir)
    if metafile == []:
        return 0
    
    metadata = lower_keys(load_json(metafile))
    
    gantry_position = parse_metadata(metadata)
    
    for i in range(0,32):
        plotNum = field_2_plot_for_season_two(gantry_position[0], i+1)
        
        src_path = os.path.join(in_dir, str(i)+'.png')
        
        if not os.path.exists(src_path):
            continue
        
        dst_dir = os.path.join(out_dir, str(plotNum))
        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)
        dst_path = os.path.join(dst_dir, scan_date+'.png')
        
        if os.path.exists(dst_path):
            continue
        
        shutil.copy(src_path, dst_path)
    
    
    return

def field_2_plot_for_season_two(x_position, y_row):

    xRange = 0
    count = 0
        
    for (xmin, xmax) in terra_common._x_range_s2:
        count = count + 1
        if (x_position > xmin) and (x_position <= xmax):
            xRange = 55 - count
            
            rows = int((y_row+1)/2)
            plotNum = convt.fieldPartition_to_plotNum(xRange, rows)
            
            return plotNum
    
    return 0

def fraction_main(in_dir):
    
    centers = get_scan_center(in_dir)
    
    #gen_field_grid(in_dir, centers)
    
    fraction_of_measured(in_dir, centers)
    
    return

def load_npy_file(in_dir):
    
    hist_list = []
    DateHist = []
    list_dirs = os.walk(in_dir)
    
    for root, sub_dir, files in list_dirs:
        for d in sub_dir:
            full_path = os.path.join(in_dir, d)
            if not os.path.isdir(full_path):
                continue
            
            histfiles = os.path.join(full_path, 'heightHist.npy')
            if not os.path.exists(histfiles):
                continue
            
            one_day_hist = np.load(histfiles, 'r')
            date_str = full_path[-10:]
            hist_list.append(one_day_hist)
            DateHist.append(date_str)
    
    return hist_list, DateHist

def load_npy_from_1728_to_864(in_dir):
    hist_list = []
    DateHist = []
    
    list_dirs = os.walk(in_dir)
    
    for root, sub_dir, files in list_dirs:
        for d in sub_dir:
            full_path = os.path.join(in_dir, d)
            if not os.path.isdir(full_path):
                continue
            
            histfiles = os.path.join(full_path, 'heightHist.npy')
            if not os.path.exists(histfiles):
                continue
            
            one_day_hist = np.load(histfiles, 'r')
            date_str = full_path[-10:]
            np_hist = np.zeros((864,400))
            for i in range(864):
                np_hist[i,:] = one_day_hist[i*2, :]+one_day_hist[i*2+1, :]
                
            hist_list.append(np_hist)
            DateHist.append(date_str)
    
    return hist_list, DateHist

def gen_plot_heatmap(in_dir, out_dir):
    
    heightlist, Datelist = load_npy_from_1728_to_864(in_dir)
    
    day_len = len(Datelist)
    
    for i in range(1, 865):
        plotHist = np.zeros((day_len, 400))
        for j in range(day_len):
            if heightlist[j][i-1,:].max() == 0:
                plotHist[j,:] = np.nan
            else:
                plotHist[j,75:] = heightlist[j][i-1,75:]
            
        draw_heatmap(plotHist, Datelist, i, out_dir)
    
    
    return

def gen_plot_Xth_percentile(in_dir, out_dir):
    
    heightlist, Datelist = load_npy_from_1728_to_864(in_dir)
    
    day_len = len(Datelist)
    
    txt_file = '/Users/nijiang/Desktop/str_date.txt'
    
    text_file = open(txt_file, "w")
    nid = 0
    
    i = 761
    percentiles = []
    for j in range(day_len):
        if heightlist[j][i-1,:].max()==0:
            continue
        else:
            localHist = heightlist[j][i-1, :]
            targetHist = localHist/np.sum(localHist)
            quantiles = np.cumsum(targetHist)
            b=np.arange(len(quantiles))
            c=b[quantiles>0.99]
            quantile_95 = min(c)
            d=b[quantiles<=0]
            if len(d) == 0:
                continue
            quantile_0 = max(d)
            estHeight = quantile_95 - quantile_0
            percentiles.append(estHeight)
            nid = nid + 1
            text_file.write("%d " % nid)
            text_file.write("%s\n" % Datelist[j])
    
    draw_percentile_plot(i, percentiles, out_dir)
    text_file.close()
    
    return

def draw_percentile_plot(plotNum, per_data, out_dir):
    
    x = np.arange(len(per_data))
    y = per_data
    
    plt.plot(x, y, 'ro')
    plt_title = 'Plot Number:%d' % plotNum
    plt.xlabel("Day")
    plt.ylabel("99th percentile height")
    plt.title(plt_title)
    
    out_file = os.path.join(out_dir, '99th_'+str(plotNum)+'.png')
    plt.savefig(out_file)
    plt.close()
    
    return

def gen_csv_file(in_dir, out_dir):
    
    heightlist, Datelist = load_npy_from_1728_to_864(in_dir)
    
    day_len = len(Datelist)
    
    for i in range(day_len):
        oneDayHist = heightlist[i]
        str_date = Datelist[i]
        out_file = os.path.join(out_dir, str_date+'_95th_quantile_height.csv')
        csv = open(out_file, 'w')
    
        (fields, traits) = get_traits_table()
        
        csv.write(','.join(map(str, fields)) + '\n')
        
        for j in range(1, 865):
            plotNum = j
            targetHist = oneDayHist[j-1,:]
            if targetHist.max() == 0:
                continue
            else:
                targetHist = targetHist/np.sum(targetHist)
                quantiles = np.cumsum(targetHist)
                b=np.arange(len(quantiles))
                c=b[quantiles>0.95]
                quantile_95 = min(c)
                d=b[quantiles<=0]
                if len(d) == 0:
                    continue
                quantile_0 = max(d)
                estHeight = quantile_95 - quantile_0
                
                str_time = str_date+'T12:00:00'
                traits['local_datetime'] = str_time
                traits['95th_quantile_height'] = str(float(estHeight)/100)
                traits['site'] = 'MAC Field Scanner Field Plot '+ str(plotNum) + ' Season 2'
                trait_list = generate_traits_list(traits)
                csv.write(','.join(map(str, trait_list)) + '\n')
    
        csv.close()
        submitToBety(out_file)
    
    
    return

def insert_height_traits_into_betydb(in_dir, out_dir):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    heightlist, Datelist = load_npy_file(in_dir)
    
    day_len = len(Datelist)
    
    for i in range(day_len):
        oneDayHist = heightlist[i]
        str_date = Datelist[i]
        out_file = os.path.join(out_dir, str_date+'_height.csv')
        csv = open(out_file, 'w')
    
        (fields, traits) = get_traits_table_height()
        
        csv.write(','.join(map(str, fields)) + '\n')
        
        for j in range(1, 1729):
            plotNum = j
            targetHist = oneDayHist[j-1,:]
            if targetHist.max() == 0:
                continue
            else:
                targetHist = targetHist/np.sum(targetHist)
                quantiles = np.cumsum(targetHist)
                b=np.arange(len(quantiles))
                c=b[quantiles>0.89]
                quantile_89 = min(c)
                d=b[quantiles<=0]
                if len(d) == 0:
                    continue
                quantile_0 = max(d)
                estHeight = quantile_89 - quantile_0
                
                str_time = str_date+'T12:00:00'
                traits['local_datetime'] = str_time
                #traits['89th_quantile_height'] = str(float(estHeight)/100)
                traits['height'] = str(F_SLOPE*float(estHeight) + F_OFFSET)
                traits['site'] = 'MAC Field Scanner Field Plot '+ str(plotNum) + ' Season 2'
                trait_list = generate_traits_list_height(traits)
                csv.write(','.join(map(str, trait_list)) + '\n')
    
    
        csv.close()
        #submitToBety(out_file)
    
    return

# BETYdb instance information for submitting output CSV (skipped if betyAPI is empty)


def submitToBety(csvfile):
    betyAPI = "https://terraref.ncsa.illinois.edu/bety/api/beta/traits.csv"
    betyKey = "D49SRRPIPFhJIiJ9XOlACRlc0BHhO3kzUJrnUBS2"

    if betyAPI != "":
        sess = requests.Session()
        print(csvfile)
        print("%s?key=%s" % (betyAPI, betyKey))
        r = sess.post("%s?key=%s" % (betyAPI, betyKey),  data=file(csvfile, 'rb').read(), headers={'Content-type': 'text/csv'})

        if r.status_code == 200 or r.status_code == 201:
            print("CSV successfully uploaded to BETYdb.")
        else:
            print("Error uploading CSV to BETYdb %s" % r.status_code)
            print(r.text)

def draw_heatmap(data, date_record, plotNum, out_dir):
    
    fig, ax = plt.subplots(figsize=(6, 4))
    data = data.transpose(1,0)
    #heatmap = ax.pcolor(data, cmap=plt.get_cmap('hot'))
    
    masked_array = np.ma.array (data, mask=np.isnan(data))
    heatmap = ax.pcolor(masked_array, cmap=plt.get_cmap('hot'))
    
    plt_title = 'Plot Number:%d' % plotNum
    plt.xlabel("Day")
    plt.ylabel("Height Level")
    plt.title(plt_title)
    plt.colorbar(heatmap)
    
    #ax.set_xticklabels(date_record, fontsize=8)
    
    out_file = 'heat_%d.png' % plotNum
    out_file = os.path.join(out_dir, out_file)
    plt.savefig(out_file)
    plt.close()
    
    return

def plot_fraction(frac_data, date_record, plotNum, out_dir):
    
    fig, ax = plt.subplots()
    x = np.arange(frac_data.size)
    
    for i in range(frac_data.size):
        plt.bar(x[i], frac_data[i], width=0.8, bottom=None, hold=None, data=None)
    
    plt_title = 'Plot Number:%d' % plotNum
    plt.xlabel("Day")
    plt.ylabel("Count of Scan")
    plt.title(plt_title)
    
    out_file = '%d_frac.png' % plotNum
    out_file = os.path.join(out_dir, out_file)
    plt.savefig(out_file)

    
    return

def fraction_of_measured(in_dir, centers):
    
    fraction = np.zeros(864)
    
    for center in centers:
        
        sub = frac_in_one_scan(center)
        
        fraction = fraction + sub
        
    
    fracPath = os.path.join(in_dir, 'fraction.npy')
    np.save(fracPath, fraction)
    
    return fraction

def frac_in_one_scan(center):
    
    frac = np.zeros(864)
    
    field_range = field_to_range(center)
    
    for i in range(16):
        plotNum = field_to_plot(field_range, i+1)
        frac[plotNum] = frac[plotNum] + 1
    
    return frac


def get_scan_center(in_dir):
    
    centers = []
    
    list_dirs = os.walk(in_dir)
    
    for root, dirs, files in list_dirs:
        for d in dirs:
            full_path = os.path.join(in_dir, d)
            if not os.path.isdir(full_path):
                continue
    
            x_center = scan_center(full_path)
            if (x_center != 0) :
                centers.append(x_center)
    
    
    return centers

def scan_center(sub_dir):
    
    metafile = find_files(sub_dir)
    if metafile == []:
        return 0
    '''
    plydata = PlyData.read(plyfile)
    if plydata.elements[0].count == 0:
        return 0
    '''
    
    metadata = lower_keys(load_json(metafile))
    
    center_position = parse_metadata(metadata)    
    
    return center_position[0]

def parse_metadata(metadata):
    
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        gantry_x = gantry_meta["position x [m]"]
        gantry_y = gantry_meta["position y [m]"]
        gantry_z = gantry_meta["position z [m]"]

    except KeyError as err:
        fail('Metadata file missing key: ' + err.args[0])
        
    position = [float(gantry_x), float(gantry_y), float(gantry_z)]
    
    return position

def load_json(meta_path):
    try:
        with open(meta_path, 'r') as fin:
            return json.load(fin)
    except Exception as ex:
        fail('Corrupt metadata file, ' + str(ex))
    
    
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

def find_files(in_dir):
    json_suffix = os.path.join(in_dir, '*_metadata.json')
    jsons = glob(json_suffix)
    if len(jsons) == 0:
        print in_dir
        fail('Could not find .json file')
        return []
    
    
    return jsons[0]

'''

def gen_field_grid(in_dir, centers):
    
    img = np.ones((540, 320), np.uint8) * 255
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
        
    for center in centers:
        center_line = (540 - (center/216*540))
        y1 = int(center_line)
        if center_line - y1 > 0.5:
            y2 = y1 + 1
        else:
            y2 = y1 - 1
        
        cv2.line(img, (0, y1), (319, y1), 128, 1)
        cv2.line(img, (0, y2), (319, y2), 128, 1)
        
    for i in range(54):
        ind = (i + 1) * 10 - 1
        ind_y = (i + 1) * 20 - 1
        
        cv2.line(img, (0, ind), (319, ind), 0, 1)
        if i < 16:
            cv2.line(img, (ind_y, 0), (ind_y, 539), 0, 1)
        
    for i in range(864):
        text = '%d' % (i+1)
        x, y = plot_to_range(i+1)
        
        x,y = range_to_coord(x, y)
        cv2.putText(img,text,(x,y), font, 0.3,0,1)
        
    file_path = 'field_scan_grid_%s.png' % in_dir[-10:]
    imgpath = os.path.join(in_dir, file_path)
    cv2.imwrite(imgpath, img)
    
    return img

'''

def range_to_coord(x, y):
    
    coor_x = (y-1) * 20
    
    coor_y = (55 - x) * 10 
    
    return coor_x, coor_y

def plot_to_range(plotNum):
    
    x = int(plotNum / 16) + 1
    y = plotNum % 16
    
    if y == 0:
        x = x - 1
        y = 16
    
    if (x % 2 == 0):
        y = 17 - y    
    
    return x, y

def field_to_range(x):
    
    return int(x / 4) + 1


def field_to_plot(plot_range, plot_row):
    if plot_range == 0:
        return 0
    
    if plot_range % 2 == 0:
        plot_row = 17 - plot_row
        
    plotNum = (plot_range-1)*16 + plot_row
    
    return plotNum

def get_traits_table():
    # Compiled traits table
    fields = ('local_datetime', '95th_quantile_height', 'access_level', 'species', 'site',
              'citation_author', 'citation_year', 'citation_title', 'method')
    traits = {'local_datetime' : '',
              '95th_quantile_height' : [],
              'access_level': '2',
              'species': 'Sorghum bicolor',
              'site': [],
              'citation_author': '"Zongyang, Li"',
              'citation_year': '2016',
              'citation_title': 'Maricopa Field Station Data and Metadata',
              'method': '95th quantiles height Estimation from 3D Scanner'}

    return (fields, traits)

def generate_traits_list(traits):
    # compose the summary traits
    trait_list = [  traits['local_datetime'],
                    traits['height'],
                    traits['access_level'],
                    traits['species'],
                    traits['site'],
                    traits['citation_author'],
                    traits['citation_year'],
                    traits['citation_title'],
                    traits['method']
                ]

    return trait_list

def get_traits_table_height():
    
    fields = ('local_datetime', 'height', 'access_level', 'species', 'site',
              'citation_author', 'citation_year', 'citation_title', 'method')
    traits = {'local_datetime' : '',
              'height' : [],
              'access_level': '2',
              'species': 'Sorghum bicolor',
              'site': [],
              'citation_author': '"Zongyang, Li"',
              'citation_year': '2016',
              'citation_title': 'Maricopa Field Station Data and Metadata',
              'method': 'height Estimation from 3D Scanner using formula: [hand height] = 28.2cm + 0.661 * [89th height percentile]'}

    return (fields, traits)

def generate_traits_list_height(traits):
    # compose the summary traits
    trait_list = [  traits['local_datetime'],
                    traits['height'],
                    traits['access_level'],
                    traits['species'],
                    traits['site'],
                    traits['citation_author'],
                    traits['citation_year'],
                    traits['citation_title'],
                    traits['method']
                ]

    return trait_list


def fail(reason):
    print >> sys.stderr, reason

if __name__ == "__main__":

    main()
