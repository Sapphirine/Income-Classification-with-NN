import gzip
import os
import numpy as np
import six
from six.moves.urllib import request


def load_matrix():
    namez=('age', 'workclass', 'fnlwgt', 'education', 'educationalNum', 'martialStatus', 'occupation', \
           'relationship', 'race', 'gender', 'capitalGain', 'capitalLoss', 'hoursPerWeek', \
           'nativeCountry', 'income')
    raw=np.genfromtxt('/home/ubuntu/FinalProject/Dataset.data', delimiter=' ',dtype=None, names=namez)
    
    num=len(raw)
    dim=14
    data = np.zeros(num*dim, dtype=np.float).reshape((num, dim))
    target= np.zeros(num, dtype=np.uint8).reshape((num, ))


    for nn in range(len(namez)-1):
        name=namez[nn]
	if type(raw[name][1])==type(raw['workclass'][1]):
	    unique=list(set(raw[name]))
	    for ll in range(len(raw)):
		line=raw[ll]
	        for hh in range(len(unique)):
	    	    if line[name]==unique[hh]:
			data[ll, nn]=hh

	
    for ii in range(len(raw)):
	tmp=raw[ii]
	if tmp[-1]=='>50K':
	    target[ii]=1
	for jj in range(len(tmp)):
 	    if type(tmp[jj]) != type(raw['workclass'][1]):
		data[ii, jj]=tmp[jj]

    return data, target


def load_data():
    if not os.path.exists('raw_data.pkl'):
	data_test, target_test = load_matrix()
    	mnist = {'data': data_test,
             'target': target_test}
    	print('Save output...')
    	with open('raw_data.pkl', 'wb') as output:
             six.moves.cPickle.dump(mnist, output, -1)
    	print('Done')

    with open('raw_data.pkl', 'rb') as raw_data_pickle:
        raw_data = six.moves.cPickle.load(raw_data_pickle)
    return raw_data
