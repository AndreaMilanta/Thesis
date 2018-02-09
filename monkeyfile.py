"""Library to save, handle and open custom data files related to monkeys
"""
import csv
import geometry as geo


def path_to_csv(path, id, datetime0, dt, filename=None):
    """write a one-day path to csv file

    Arguments:
        path {List[List[Coordinates], List[Coordinates]]} -- list of points
        id {int} -- id of monkey
        datetime0 {datetime} -- date and time of first point
        dt {int} -- distance in seconds between datapoints

    Keyword Arguments:
        filename {string} -- name of output file. If None the file is saved as id_date.csv (default: {None})
    """
# Control File Preparation
    linecount = 0
    content = []
    fullpath = []                       # Unique full path of monkey on date
    for p in path:
        bgn = linecount
        fullpath.extend(p[0])           # add first part (random)
        if p[1] is None:
            end = linecount + len(p[0]) - 1
            # vwp = None
            vwp = 'None'
        else:
            vwp = linecount + len(p[0]) - 1
            end = vwp + len(p[1])
            fullpath.extend(p[1])       # add second part
        linecount = end + 1
        row = (bgn, vwp, end)
        content.append(row)
# Filename preparation
    if filename is None:
        filename = str(id) + '_' + str(datetime0.date()).replace('-', '')
    filename = filename.split('.')[0] + '.csv'              # Force filetype to .csv
    fname_info = filename.split('.')[0] + '_info.csv'
# Write info file
    header_info = ['start', 'viewpoint', 'end']
    with open(fname_info, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header_info)
        writer.writerows(content)
        # for row in content:
        #     writer.writerow(row)
# Write data file
    df = geo.getDataframe(fullpath, datetime0, dt)
    df.to_csv(filename, na_rep="None", float_format='%.3f', header=True, index=False)
