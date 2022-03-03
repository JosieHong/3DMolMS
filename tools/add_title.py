'''
Date: 2021-02-19 15:24:06
LastEditors: yuhhong
LastEditTime: 2021-11-24 18:02:29
Note: 
    python add_title.py <path to .mfg file> <path to output .mgf file>
    python add_title.py ../data/GNPS/ALL_GNPS.mgf ../data/GNPS/ALL_GNPS_titled.mgf
'''
import sys

MGF_FILE = sys.argv[1]
OUT_FILE = sys.argv[2]

with open(MGF_FILE, "r") as f:
    data = f.readlines()

output = ""
idx = 0
prefix = MGF_FILE.split('/')[-1]
for d in data: 
    if d == "BEGIN IONS\n":
        output += d + "TITLE="+prefix+'.'+str(idx)+"\n"
        idx += 1
    else:
        output += d

with open(OUT_FILE, "w") as f:
    f.write(output)