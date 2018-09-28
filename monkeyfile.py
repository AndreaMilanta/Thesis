"""Library to save, handle and open custom data files related to monkeys
"""
import csv


def path_to_csv(date, folderpath, filename=None):
    """write a one-day path to csv file

    Arguments:
        datepath {datepath} -- list of points

    Keyword Arguments:
        filename {string} -- name of output file. If None the file is saved as id_date.csv (default: {None})
    """
# Control File Preparation
    linecount = 0
    content = []
    fullpath = []                       # Unique full path of monkey on date
    for seg in date.path():
        bgn = linecount
        fullpath.extend(seg[0])           # add first part (random)
        if not(seg[0]):
            end = linecount + len(seg[1]) - 1
            vwp = 'None'
        else:
            vwp = linecount + len(seg[0]) - 1
            end = vwp + len(seg[1])
            fullpath.extend(seg[1])       # add second part
        linecount = end + 1
        row = (bgn, vwp, end)
        content.append(row)
# Filename preparation
    if filename is None:
        fname = str(date.id) + '_' + str(date.date).replace('-', '')
    filename = folderpath + fname.split('.')[0] + '.csv'              # Force filetype to .csv
    fname_info = folderpath + fname.split('.')[0] + '_info.csv'
# Write info file
    header_info = ['start', 'viewpoint', 'end']
    with open(fname_info, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header_info)
        writer.writerows(content)
        # for row in content:
        #     writer.writerow(row)
# Write data file
    df = date.toDataframe()
    df.to_csv(filename, na_rep='None', float_format='%.3f', header=True, index=False)
