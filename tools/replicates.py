'''
Date: 2022-06-27 16:33:00
LastEditors: yuhhong
LastEditTime: 2022-06-27 16:33:00
'''
import sys
import argparse
from tqdm import tqdm

import re
from pyteomics import mgf
import pandas as pd
from decimal import *
import numpy as np
from numpy import dot
from numpy.linalg import norm

ENCODE_CE = {
			# HMDB
			'low': 10, 'med': 20, 'high': 40}

def parse2ce(ce_str, pepmass, charge=2): 
	ce = None
	if ce_str == 'unknown': # give the unknown collision energy an average value
		ce = 20
	elif ce_str in ENCODE_CE.keys(): 
		ce = ENCODE_CE[ce_str]
	else: # parase ce for NIST20 & MassBank
		# match collision energy (eV)
		matches_ev = {
			# NIST20
			r"^[\d]+[.]?[\d]*$": lambda x: float(x), 
			r"^[\d]+[.]?[\d]*[ ]?eV$": lambda x: float(x.rstrip(" eV")), 
			r"^[\d]+[.]?[\d]*[ ]?ev$": lambda x: float(x.rstrip(" ev")), 
			r"^[\d]+[.]?[\d]*[ ]?v$": lambda x: float(x.rstrip(" v")), 
			r"^NCE=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*eV$": lambda x: float(x.split()[1].rstrip("eV")),
			# MassBank
			r"^[\d]+[.]?[\d]*[ ]?v$": lambda x: float(x.rstrip(" v")), 
			r"^ramp [\d]+[.]?[\d]*-[\d]+[.]?[\d]* (ev|v)$":  lambda x: float((float(re.split(' |-', x)[1]) + float(re.split(' |-', x)[2])) /2), 
			r"^[\d]+[.]?[\d]*-[\d]+[.]?[\d]*$": lambda x: float((float(x.split('-')[0]) + float(x.split('-')[1])) /2), 
			r"^hcd[\d]+[.]?[\d]*$": lambda x: float(x.lstrip('hcd')), 
		}
		for k, v in matches_ev.items(): 
			if re.match(k, ce_str): 
				ce = v(ce_str)
				break
	if ce == None: 
		ce = 0
	return ce

def cosine_similarity(list_1, list_2):
	cos_sim = dot(list_1, list_2) / (norm(list_1) * norm(list_2))
	return cos_sim

def ms2vec(x, y, pepmass, resolution=0.2, num_ms=2000): 
	pepmass = Decimal(str(pepmass)) 
	resolution = Decimal(str(resolution))
	right_bound = int(pepmass // resolution)

	ms = [0] * (int(Decimal(str(num_ms)) // resolution)) # add "0" to y data
	for idx, val in enumerate(x): 
		val = int(round(Decimal(str(val)) // resolution))
		if val >= right_bound: 
			continue
		ms[val] += float(y[idx])
	return ms

def vec2ms(ms, resolution=0.2): 
	x = []
	y = []
	for i, j in enumerate(ms): 
		if j != 0: 
			x.append(float(i*resolution))
			y.append(float(j))
	return x, y

def intra_cs(df): 
	if len(df) > 1: 
		mol_df = {'clean_id': [], 'smiles': [], 
					'mass': [], 'collision_energy': [], 'precursor_type': [], 
					'avg_ms_mz': [], 'avg_ms_intensity': [], 
					'intra_cs': []}
		mol_df['clean_id'].append(";".join(df['clean_id'].tolist()))
		mol_df['smiles'].append(df.iloc[0]['smiles'])
		mass = df.iloc[0]['mass']
		mol_df['mass'].append(mass)
		mol_df['collision_energy'].append(df.iloc[0]['collision_energy'])
		mol_df['precursor_type'].append(df.iloc[0]['precursor_type'])

		all_vectors = []
		for x, y in zip(df['mz'].tolist(), df['intensity'].tolist()): 
			all_vectors.append(ms2vec(x, y, mass))
		avg_vector = np.mean(all_vectors, axis=0)

		# save the average spectra, which will be used in 
		# calculating inter-cosine similarity
		avg_mz, avg_intensity = vec2ms(avg_vector)
		mol_df['avg_ms_mz'].append(avg_mz)
		mol_df['avg_ms_intensity'].append(avg_intensity)

		# intra-cosine similarity
		all_cosine_sim = []
		for vector in all_vectors:
			cs = cosine_similarity(vector, avg_vector)
			if cs == np.nan:
				continue    
			all_cosine_sim.append(cs)
		mol_df['intra_cs'].append(np.mean(all_cosine_sim, axis=0))
		mol_df = pd.DataFrame.from_dict(mol_df)
	else: 
		# mol_df = df.drop(columns=['mz', 'intensity'])
		mol_df = df.rename(columns={"mz": "avg_ms_mz", "intensity": "avg_ms_intensity"})
		mol_df['intra_cs'] = 1.0
	return mol_df

def inter_cs(df): 
	if len(df) == 1:
		df['inter_cs'] = None
	elif len(df==2): 
		ms1 = ms2vec(df.iloc[[0]]['avg_ms_mz'].tolist()[0], df.iloc[[0]]['avg_ms_intensity'].tolist()[0], mass)
		ms2 = ms2vec(df.iloc[[1]]['avg_ms_mz'].tolist()[0], df.iloc[[1]]['avg_ms_intensity'].tolist()[0], mass)
		df['inter_cs'] = cosine_similarity(ms1, ms2)
	else:
		raise Exception('Do not implement inter cosine similarity among 3 sets.')
	return df.drop(columns=['avg_ms_mz', 'avg_ms_intensity', 'mass'])



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Preprocess the Data')
	parser.add_argument('--input1', type=str, default = '',
						help='path to input data')
	parser.add_argument('--input2', type=str, default = '',
						help='path to input data')
	parser.add_argument('--output', type=str, default = '',
						help='path to input data')
	parser.add_argument('--ion_mode', type=str, default='ALL', choices=['P', 'N', 'ALL'], 
						help='Ion mode used for training and test') 
	args = parser.parse_args()

	# convert mgf to csv
	input_mgf1 = args.input1
	input_mgf2 = args.input2
	output_csv = args.output

	if args.ion_mode == 'P':
		ion_mode = ['p', 'positive']
	elif args.ion_mode == 'N':
		ion_mode = ['n', 'negative']
	else:
		ion_mode = ['p', 'positive', 'n', 'negative']

	# load the first input
	with mgf.read(input_mgf1) as reader: 
		print("Got {} data from {}".format(len(reader), input_mgf1))
		df1 = {}
		for idx, spectrum in enumerate(tqdm(reader)): 
			if spectrum['params']['ionmode'].lower() not in ion_mode: 
				continue

			clean_id = spectrum['params']['clean_id']
			smiles = spectrum['params']['smiles']
			mass = spectrum['params']['pepmass'][0]
			collision_energy = parse2ce(spectrum['params']['collision_energy'].lower(), mass) # j0siee: this is ce, rather than nce
			precursor_type = spectrum['params']['precursor_type']
			mz = spectrum['m/z array'].tolist()
			intensity = spectrum['intensity array'].tolist()
			df1[idx] = [clean_id, smiles, mass, collision_energy, precursor_type, mz, intensity]
		df1 = pd.DataFrame.from_dict(df1, orient='index', 
									columns=['clean_id', 'smiles', 'mass', 'collision_energy', 'precursor_type', 'mz', 'intensity'])
		print(df1)

	# load the second input
	with mgf.read(input_mgf2) as reader: 
		print("Got {} data from {}".format(len(reader), input_mgf2))
		df2 = {}
		for idx, spectrum in enumerate(tqdm(reader)): 
			if spectrum['params']['ionmode'].lower() not in ion_mode: 
				continue

			clean_id = spectrum['params']['clean_id']
			smiles = spectrum['params']['smiles']
			mass = spectrum['params']['pepmass'][0]
			collision_energy = parse2ce(spectrum['params']['collision_energy'].lower(), mass) # j0siee: this is ce, rather than nce
			precursor_type = spectrum['params']['precursor_type']
			mz = spectrum['m/z array'].tolist()
			intensity = spectrum['intensity array'].tolist()
			df2[idx] = [clean_id, smiles, mass, collision_energy, precursor_type, mz, intensity]
		df2 = pd.DataFrame.from_dict(df2, orient='index', 
									columns=['clean_id', 'smiles', 'mass', 'collision_energy', 'precursor_type', 'mz', 'intensity'])
		print(df2)

	# calculate average spectra & intra-cosine similarity
	df1['dataset'] = 'dataset1'
	df1 = df1.groupby(by=['smiles', 'collision_energy', 'precursor_type']).apply(intra_cs).reset_index(drop=True)

	df2['dataset'] = 'dataset2'
	df2 = df2.groupby(by=['smiles', 'collision_energy', 'precursor_type']).apply(intra_cs).reset_index(drop=True)

	df = pd.concat([df1, df2], ignore_index=True)

	# j0siee: print overlap compound-spectra
	# df1 = df[df['dataset']=='agilent'].drop(columns=['avg_ms_mz', 'avg_ms_intensity', 'mass'])
	# df2 = df[df['dataset']=='nist'].drop(columns=['avg_ms_mz', 'avg_ms_intensity', 'mass'])
	# tmp = df1.merge(df2, on=['smiles', 'precursor_type'], how='inner')
	# print(tmp)
	# tmp.to_csv('../tmp.csv', sep='\t')
	# exit()

	# calculate inter-cosine similarity
	inter_df = df.drop_duplicates(subset=['dataset', 'smiles', 'collision_energy', 'precursor_type'])
	inter_df = inter_df.groupby(by=['smiles', 'collision_energy', 'precursor_type']).apply(inter_cs).reset_index(drop=True)
	inter_df = inter_df.dropna()
	print(inter_df)
	print(inter_df.columns)

	print('Save the inter-cosine similarity results.')
	inter_df.to_csv(output_csv, sep='\t') # save the replicated results

	# output the results
	print('Intra-cosine similarity in datasets:')
	print(df.groupby(by=['dataset'])['intra_cs'].mean())
	print(df.groupby(by=['dataset'])['intra_cs'].std())
	print(df.groupby(by=['dataset'])['intra_cs'].count())

	print('\nInter-cosine similarity between datasets:')
	print(inter_df['inter_cs'].mean())
	print(inter_df['inter_cs'].std())
	print(inter_df['inter_cs'].count())



'''
Inter-cosine similarity between datasets:
0.9465120124814748
0.14486023614923477
1472
'''