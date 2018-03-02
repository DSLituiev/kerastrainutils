
# coding: utf-8
import sys
import os
from functools import reduce
from os.path import join as pjoin
from scandir import scandir
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import pandas as pd


def get_code_dict():
    CODE_TEXT = """Chain code value	0	1	2	3	4	5	6	7
    X Coordinate	0	1	1	1	0	-1	-1	-1
    Y coordinate	-1	-1	0	1	1	1	0	-1"""
    key, xval, yval =[[int(y) for y in x.split("\t")[1:]] for x in  CODE_TEXT.split("\n")]
    CODE_DICT = dict(zip(key, list(zip(xval, yval))))
    return CODE_DICT
CODE_DICT = get_code_dict()


def find_overlays(parentdir, pair_type = "png"):
    """
    pair_type = "LJPEG"
    """
    fbases_w_overlay = [".".join(x.split(".")[:-1]) for x in os.listdir(parentdir) if x.endswith("OVERLAY")]

    for fb in fbases_w_overlay:
        if not (os.path.isfile(pjoin(parentdir, fb+".OVERLAY")) and 
           os.path.isfile(pjoin(parentdir, fb+"." + pair_type))):
            fbases_w_overlay.remove(fb)
            print("pair not found for:", fb)
    return fbases_w_overlay

def find_parse_meta(parentdir):
    fn_meta = [x.name for x in scandir(parentdir) if x.name.endswith(".ics")]
    if len(fn_meta) > 1:
        print(fn_meta)
        raise Exception("more than one meta file in {}".format(parentdir))
    if len(fn_meta) == 0:
        raise Exception("no meta file found in {}".format(parentdir))
    fn_meta = fn_meta[0]
    fp_meta = pjoin(parentdir, fn_meta)
    #print(fp_meta)
    return parse_meta(fp_meta)


def parse_meta(fp_meta):
    """reads an *.ics file and produces a pandas.DataFrame with fields:
    (index: view)
    width       (int)
    height      (int)
    resolution  (int)
    overlay     (bool)
    """
    df_meta = []
    startreading = False
    with open(fp_meta, 'r') as fh:
        for line in fh:
            if startreading:
                try:
                    line = parse_meta_descr_line(line)
                except:
                    print("="*20)
                    print("parsing failed in {}".format(fp_meta))
                    print(df_meta.columns)
                    pass
                df_meta.append(line)
            if "SEQUENCE" in line:
                startreading = True
    
    df_meta = pd.DataFrame(df_meta)
    df_meta.set_index("view", inplace=True)
    df_meta.rename_axis({"pixels_per_line":"width", "lines":"height"}, axis=1, inplace=True)
    return df_meta


def parse_meta_descr_line(line):
    prevcol = ""
    line = line.split(" ")
    view_dict = {"view": line[0], "overlay":line[-1].startswith("OV")}
    for col in line[1:-1]:
        if prevcol in ("LINES", "PIXELS_PER_LINE"):
            try:
                view_dict[prevcol.lower()] = int(col)
            except:
                view_dict[prevcol.lower()] = col
        elif prevcol == "RESOLUTION":
            try:
                view_dict[prevcol.lower()] = float(col)
            except:
                view_dict[prevcol.lower()] = col
            
        prevcol = col
    return view_dict


def parse_overlay_file(fp_overlay, noroi = False):
    """reads and parses *.OVERLAY files from DDSM project
    Output:
    list of dictionaries
    within each list, the outlines can be found 
    under 'BOUNDARY' and/or 'CORE' keys
    """
    rois = []
    startreading = False
    curr_roi = {}
    prev_key = ""
    with open(fp_overlay, 'r') as fh:
        tot_abn = next(fh)
        #print(tot_abn)
        for line in fh:
            line = line.rstrip(" #\n")
            if line.startswith("ABNORMALITY"):
                if len(curr_roi)>0:
                    rois.append(curr_roi)
    #                 print("appending")
                curr_roi = {}
            line = line.split(" ")
            if len(line[0])==0:
                print(line) ###########
            elif not line[0][0].isdigit():
                if len(line)>1:
                    if line[0] in ["ASSESSMENT", "SUBTLETY"]:
                        curr_roi[line[0]] = int(line[1])
                    else:
                        curr_roi[line[0]] = " ".join(line[1:])
                    #print(line)
            else:
                prev_key = prev_key.lstrip('\t')
    #             print(prev_key)
                if not noroi:
                    curr_roi[prev_key] = parse_roi_line(line)
                else:
                    curr_roi[prev_key] = len(line) - 2
            prev_key = line[0]

    rois.append(curr_roi)
    return rois


def parse_roi_line(roi):
    if type(roi) is str:
        roi = roi.rstrip(" #\n")
        roi = roi.split(" ")
    elif type(roi) is list:
        pass
    else:
        raise ValueError("unknown input type")
        
    roi = [int(x) for x in roi if x!=""]
    #
    x0, y0 = roi[:2]
    roi_coords = [(x0, y0)]
    #
    roi = roi[2:]
    #
    for kk in roi:
        dx, dy = CODE_DICT[kk]
        xprev, yprev = roi_coords[-1]
        roi_coords.append((xprev+dx, yprev+dy))
    return roi_coords


def get_roi_mask(roi, width, height, fill=1):
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(roi, outline=fill, fill=fill)
    mask = np.asarray(img, dtype='uint8')
    return mask


def get_channel_num(ds):
    if ds["LESION_TYPE"].startswith("MASS"):
        return (ds["ASSESSMENT"]-1)
    elif ds["LESION_TYPE"].startswith("CALCIFICATION"):
        return 4+(ds["ASSESSMENT"])
    else:
        return -1

def get_channel_simple5(ds):
    if ds["LESION_TYPE"].startswith("MASS"):
        if ds["PATHOLOGY"] == "MALIGNANT":
            return 1
        else:
            return 3
    elif ds["LESION_TYPE"].startswith("CALCIFICATION"):
        if ds["PATHOLOGY"] == "MALIGNANT":
            return 2
        else:
            return 4

def get_roi_stats(parentdir, noroi=True,
                  savemask=False,
                  channelmap = {"channel_camass_birad": get_channel_simple5}):
#                  channelmap = {"channel_camass_birad": get_channel_num}):
    """get roi statistics within a case directory
    NOTE: width and height might be mixed up!!!
    """
    fbases_w_overlay = find_overlays(parentdir)
    view_base_w_overlay_dict = {ff.split(".")[-1]:ff for ff in fbases_w_overlay}
    # print(view_base_w_overlay_dict)

    df_meta = find_parse_meta(parentdir)
    df_meta_ol = df_meta[df_meta.overlay].copy()
    
#     print(df_meta_ol)
    df_meta_ol["filebase"] =  [view_base_w_overlay_dict.get(x, None) for \
                                x in df_meta_ol.index]
    df_meta_ol = df_meta_ol[~df_meta_ol["filebase"].isnull()]

    rois_dir = []
    for view, row in df_meta_ol.iterrows():
        "in each file:"
        #width, height = row["width"], row["height"]
        width  = min(row["width"], row["height"])
        height = max(row["width"], row["height"])
        try:
            fp_overlay = pjoin(parentdir, row["filebase"] + ".OVERLAY")
        except Exception as ee:
            raise ee
        rois = parse_overlay_file(fp_overlay, noroi=noroi)
        for rr in rois:
            rr["view"] = view
            for kk,ff in channelmap.items():
                rr[kk] = ff(rr)
        rois_dir.extend(rois)
        
        if savemask:
            for roi in rois:
#                 print(roi)
                masks = {}
                for linelabel in ["BOUNDARY", "CORE"]:
                    if linelabel in roi:
                        masks[linelabel] = []
                linelabels = masks.keys()
                for linelabel in linelabels:
#                     if linelabel in 
                    polygon = roi[linelabel]
                    masks[linelabel].append(
                        get_roi_mask(polygon, width, height, fill=roi["channel_camass_birad"])
                                           )
                for linelabel in linelabels:
                    if len(masks)>0:
                        masks[linelabel] = reduce(lambda x,y: x+y, masks[linelabel])
                                                  
                for linelabel in linelabels:
                    if len(masks[linelabel])==0:
                        continue
                    fn_mask = pjoin(parentdir,
                                "{}.{}.mask.png".format(row["filebase"], linelabel))
                    print("saving a mask of size {} to\n{}".format(masks[linelabel].shape, fn_mask))
                    #plt.imsave(fn_mask, masks[linelabel],)
                    masks[linelabel] = Image.fromarray(masks[linelabel])
                    masks[linelabel] = masks[linelabel].convert('L')
                    masks[linelabel].save(fn_mask, 'png')

    rois_dir = pd.DataFrame.from_dict(rois_dir)
#     for kk in ["ASSESSMENT", "SUBTLETY"]:
#         rois_dir[kk] = rois_dir[kk].astype(int)
    return rois_dir

if __name__ == '__main__':

    parentdir="/home/ubuntu/data/ddsm/figment.csee.usf.edu/pub/DDSM/cases/cancers/cancer_07/case1235/"
    roi_stats_ = get_roi_stats(parentdir, noroi=False, savemask=True)
    print(roi_stats_)

    sys.exit(0)
    superdir =  "/home/ubuntu/data/ddsm/figment.csee.usf.edu/pub/DDSM/cases/cancers/"
    roi_stats = []
    for ss in scandir(superdir):
        setdir = pjoin(superdir, ss.name)
        for dd in scandir(setdir):
            parentdir = pjoin(setdir, dd.name)
            if not os.path.isdir(parentdir):
                continue
            
            roi_stats_ = get_roi_stats(parentdir, noroi=False, savemask=True)
            roi_stats_["dir"] = parentdir
            roi_stats.append( roi_stats_ )
    roi_stats = pd.concat(roi_stats)
    print(roi_stats.shape)

