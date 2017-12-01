import argparse
from scanner_3d.plant_height import process_all_scanner_data, full_day_gen_hist, full_day_array_to_xlsx, \
    process_one_month_data


def options():

    parser = argparse.ArgumentParser(description='Height Distribution Extractor in Roger',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-m", "--mode", help="all day flag, all for all day process, "
                                             "given parent directory as input, one for one day process")
    parser.add_argument("-p", "--ply_dir", help="ply directory")
    parser.add_argument("-j", "--json_dir", help="json directory")
    parser.add_argument("-o", "--out_dir", help="output directory")
    parser.add_argument("-y", "--year", help="specify which year to process")
    parser.add_argument("-d", "--month", help="specify month")
    parser.add_argument("-s", "--start_date", help="specify start date")
    parser.add_argument("-e", "--end_date", help="specify end date")

    args = parser.parse_args()

    return args


def main():
    print("start...")

    args = options()

    if args.mode == 'all':
        process_all_scanner_data(args.ply_dir, args.json_dir, args.out_dir)

    if args.mode == 'one':
        full_day_gen_hist(args.ply_dir, args.json_dir, args.out_dir)

        full_day_array_to_xlsx(args.out_dir)

    if args.mode == 'date':
        process_one_month_data(args.ply_dir, args.json_dir, args.out_dir, args.year, args.month, args.start_date, args.end_date)

    return


if __name__ == "__main__":

    main()