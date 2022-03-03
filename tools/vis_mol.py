'''
Date: 2022-04-11 12:42:08
LastEditors: yuhhong
LastEditTime: 2022-04-11 13:24:27
'''
import numpy as np
import matplotlib.pyplot as plt

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

ENCODE_ATOM = {'C': 70, 'H': 25, 'O': 60, 'N': 65, 'F': 50, 'S': 100, 'Cl': 100, 'P': 100, 'B': 85, 'Br': 94, 'I': 140}

def molblock2set(mol_block): 
        '''
        Input:  mol_block   [list denotes the lines of mol block]
        Return: points      [numpy array denotes the atom points, (npoints, 4)]
        '''
        points = []
        for d in mol_block:
            if len(d) == 69: # the format of molecular block is fixed
                atom = [i for i in d.split(" ") if i!= ""]
                # atom: [x, y, z, atom_type, charge, stereo_care_box, valence]
                # sdf format (atom block): https://docs.chemaxon.com/display/docs/mdl-molfiles-rgfiles-sdfiles-rxnfiles-rdfiles-formats.md
                
                if len(atom) == 16 and atom[3] in ENCODE_ATOM.keys(): 
                    # only x-y-z coordinates
                    # point = [float(atom[0]), float(atom[1]), float(atom[2])]

                    # x-y-z coordinates and atom type
                    point = [float(atom[0]), float(atom[1]), float(atom[2]), ENCODE_ATOM[atom[3]]]
                    points.append(point)
                elif len(atom) == 16: # check the atom type
                    print("Error: {} is not in {}, please check the dataset.".format(atom[3], ENCODE_ATOM.keys()))
                    exit()
        
        # normalize the point cloud
        # We normalize scale to fit points in a unit sphere
        points = np.array(points)
        points_xyz = points[:, :3]
        centroid = np.mean(points_xyz, axis=0)
        points_xyz -= centroid
        points = np.concatenate((points_xyz, points[:, 3:]), axis=1)
        return points

if __name__ == '__main__':
    smiles = "CC1(C)[C@H](C(O)=O)N2[C@@H](CC2)S1"
    outpath = "../img/mol.png"

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    mol_block = Chem.MolToMolBlock(mol).split("\n")
    mol_set = molblock2set(mol_block)

    fig = plt.figure(dpi=200)
    ax = plt.axes(projection="3d")

    x_points = mol_set[:, 0]
    y_points = mol_set[:, 1]
    z_points = mol_set[:, 2]
    s_points = mol_set[:, 3]
    ax.scatter3D(x_points, y_points, z_points, c='k', s=s_points, depthshade=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # plt.show()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()
    print("Save figure to {}".format(outpath))