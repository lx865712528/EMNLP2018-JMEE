import sys

import os

args = sys.argv
if len(args) != 3:
    sys.exit(-1)

input_path = args[1]
output_path = args[2]
if not os.path.exists(output_path):
    os.makedirs(output_path)

split_names = ["training", "test", "dev"]

for split_name in split_names:
    file_name = os.path.join(output_path, split_name + ".jl")
    with open(file_name, "wb") as fout, open("qi_filelist/new_filelist_ACE_%s" % split_name, "rb") as flist:
        for fname in flist:
            fname = fname.strip()
            if len(fname) == 0: continue
            fname = fname.split("/")[-1] + ".json"
            with open(os.path.join(input_path, fname), "rb") as jf:
                jj = jf.read().strip() + "\n"
                fout.write(jj)

print("Done!")
