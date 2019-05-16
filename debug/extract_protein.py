"""
Extract information of a protein with a particular PDB ID from the raw
ProteinNet data so that it can be tested with the code to check for
errors and debugging
"""

import sys

if len(sys.argv) != 3:
    print("Usage: python extract_protein.py <proteinnet_file> <pdb_id>")
    exit(1)

file = sys.argv[1]
pdb_id = sys.argv[2]
print("File will be stored in ./data/raw/" + pdb_id)
flg = 0

with open(file, "r") as f, open("./data/raw/" + pdb_id, "w") as out:
    while True:
        line = f.readline()
        if line == "[ID]\n":
            if flg:
                break
            next_line = f.readline()
            if next_line == pdb_id + "\n":
                flg = 1
                out.write(line)
                out.write(next_line)
                line = f.readline()
        if flg:
            out.write(line)
