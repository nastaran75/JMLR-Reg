import time
import os
import sys 
from myutil import *
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
from generate_data import generate_data 
from triage_human_machine import triage_human_machine


def print_new_pca_data( data_file_old, data_file_new,n):
	data=load_data( data_file_old )
	data['X']=data['X'][:,:n]
	data['test']['X']=data['test']['X'][:,:n]
	save( data, data_file_new)

def relocate_data( data_file_old, data_file_new ):
	data=load_data( data_file_old)
	save( data, data_file_new )

class eval_triage:
	def __init__(self,data_file,real_flag=None, real_wt_std=None):
		self.data=load_data(data_file)
		self.real=real_flag		
		self.real_wt_std=real_wt_std


	def eval_loop(self,param,res_file,option):	
		res=load_data(res_file,'ifexists')		
		for std in param['std']:
			if self.real:
				data_dict=self.data
				triage_obj=triage_human_machine(data_dict,self.real)
			else:
				if self.real_wt_std:
					data_dict = {'X':self.data['X'],'Y':self.data['Y'],'c': self.data['c'][str(std)]}
					triage_obj=triage_human_machine(data_dict,self.real_wt_std)
				else:
					test={'X':self.data.Xtest,'Y':self.data.Ytest,'human_pred':self.data.human_pred_test[str(std)]}
					data_dict = {'test':test,'dist_mat':self.data.dist_mat,  'X':self.data.Xtrain,'Y':self.data.Ytrain,'human_pred':self.data.human_pred_train[str(std)]}
					triage_obj=triage_human_machine(data_dict,False)
			if str(std) not in res:
				res[str(std)]={}
			for K in param['K']:
				if str(K) not in res[str(std)]:
					res[str(std)][str(K)]={}
				for lamb in param['lamb']:
					if str(lamb) not in res[str(std)][str(K)]:
						res[str(std)][str(K)][str(lamb)]={}
					# res[str(std)][str(K)][str(lamb)]['greedy'] = triage_obj.algorithmic_triage({'K':K,'lamb':lamb},optim='greedy')
					print 'std-->', std, 'K--> ',K,' Lamb--> ',lamb
					res_dict = triage_obj.algorithmic_triage({'K':K,'lamb':lamb},optim=option)
					res[str(std)][str(K)][str(lamb)][option] = res_dict
		save(res,res_file)
def main():
	s = ['sigmoid_fig_2_n500d5','gauss_fig_2_n500d5','stare5','stare11','messidor'][1]
	if 'sigmoid' in s :
		data_file = '../Synthetic_data/data_dict_' + s
		res_file = '../Synthetic_data/res_' + s
		list_of_std = [0.01,0.02,0.03,0.04,0.05]
		list_of_lamb= [0.001]
	if 'gauss' in s:
		data_file = '../Synthetic_data/data_dict_' + s
		res_file = '../Synthetic_data/res_' + s
		list_of_std = [0.01,0.02,0.03,0.04,0.05]
		list_of_lamb= [0.005]
	if 'stare5' in s:
		list_of_std = [.2, .4, .6, .8]
		list_of_lamb = [1]
		path = '../Real_Data_Results/'
		data_file = path + 'data/' + s + '_pca50'
		res_file = path + s + '_res_pca50'

	if 'stare11' in s:
		list_of_std = [.2, .4, .6, .8]
		list_of_lamb = [1]
		path = '../Real_Data_Results/'
		data_file = path + 'data/' + s + '_pca50'
		res_file = path + s + '_res_pca50'
	if 'messidor' in s:
		list_of_std = [.2, .4, .6, .8]
		list_of_lamb = [1]
		path = '../Real_Data_Results/'
		data_file = path + 'data/' + s + '_pca50'
		res_file = path + s + '_res_pca50'

	print s
	list_of_option =['greedy']#,'diff_submod','distort_greedy','stochastic_distort_greedy','kl_triage' ]#[int(sys.argv[2])]

	obj=eval_triage(data_file,real_wt_std=True)
	for option in list_of_option:
		if option in ['diff_submod' ]:
			list_of_K = [ 0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 ]
		else:
			list_of_K = [0.99]
		param={'std':list_of_std,'K':list_of_K,'lamb':list_of_lamb}
		obj.eval_loop(param,res_file,option)

if __name__=="__main__":
	main()