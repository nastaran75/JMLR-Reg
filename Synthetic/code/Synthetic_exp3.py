import time
import os
import sys

import matplotlib
from brewer2mpl import brewer2mpl

from myutil import *
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
from generate_data import generate_data 
from triage_human_machine import triage_human_machine
from matplotlib.ticker import FormatStrFormatter


class Synthetic_exp3:
	def __init__(self,data_file,list_of_K, list_of_lamb, list_of_std):
		self.data=load_data(data_file)
		self.list_of_K = list_of_K
		self.list_of_lamb = list_of_lamb
		self.list_of_std = list_of_std

	def eval_loop(self, image_file_pre,image_file_name,option):
		for K in self.list_of_K :
			for lamb in self.list_of_lamb :
				print '\\begin{figure}[H]'		
				for std,ind  in  zip(self.list_of_std, range(len(self.list_of_std)) ):
					image_file = image_file_name + str(int(400*K))+'_new'
					# suffix =  '_' + str(std) # '_'+str(K) + '_' + str(lamb) +
					# image_file = image_file_pre + '_std_'+ str(std ) + '_lamb_'+str(lamb)+'_K_'+str(K) #suffix.replace('.','_')
					caption='$K = '+str(K)+',    \\lambda='+str(lamb)+' , \\rho = '+ str(std)+'$'
					# self.print_figure_singleton( image_file.split('/')[-1], caption )
					local_data = self.data[ str( std ) ]
					triage_obj=triage_human_machine( local_data, True )    
					res_obj =  triage_obj.algorithmic_triage({'K':K,'lamb':lamb},optim='greedy')
					subset_human = res_obj['subset']
					n=local_data['X'].shape[0]
					subset_machine = np.array( [ i for i in range(n) if i not in subset_human])
					self.plot_subset( local_data, res_obj['w'], subset_human,  subset_machine, image_file, K ,option)

				print '\\caption{'+image_file_pre.split('/')[-1].replace('_',' ')+', lamb = '+str(lamb)+' }'
				print '\\end{figure}'

	def plot_subset(self, data, w, subset_human, subset_machine, image_file, K,option):
		matplotlib.rcParams['text.usetex'] = True
		plt.rc('font', family='serif')
		plt.rc('xtick', labelsize=21)
		plt.rc('ytick', labelsize=21)
		fig, ax = plt.subplots()
		fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
		# fig.set_size_inches(7.2, 5.4)
		bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
		color_list = bmap.mpl_colors
		key = 'test'

		x = data['X'][subset_machine, 0].flatten()
		y = data['Y'][subset_machine]
		plt.scatter(x, y, c=color_list[0], label=r'$\mathcal{V}$\textbackslash $\mathcal{S}^*$')
		# self.write_to_txt( x,y, image_file+'_machine')

		x=data['X'][subset_human,0].flatten()
		y=data['Y'][subset_human]
		plt.scatter(x,y,c=color_list[1],label=r'$\mathcal{S}^*$')
		# self.write_to_txt( x,y, image_file+'_human')
		x=data['X'][:,0].flatten()
		y=data['X'].dot(w).flatten()
		plt.scatter(x,y,c=color_list[2],label='$\hat y = {w^*}^\mathsf{T}(\mathcal{S}^*) \mathbf{x}$ ')
		# self.write_to_txt( x,y, image_file+'_prediction')
		if option=='sigm':
			xlabel = '$[-7,7]$'
			ax.set_ylim([-0.5, 1.5])
			x = -5
			# if(K==0.2):
			# 	x = 15
		if option == 'gauss':
			xlabel = '$[-1,1]$'
			ax.set_ylim([0.17,0.21])
			x = 3
			# if(K==0.6 or K==0.8):
			# 	x = 7

		plt.legend(prop={'size': 19}, frameon=False,
                   handlelength=0.2)
		plt.xlabel(r'\textbf{Features} $x$ $\sim$ \textbf{Unif} '+xlabel, fontsize=23)
		ax.set_ylabel(r'\textbf{Response} $(y)$', fontsize=23,
					  labelpad=x)  # ('MSE' +r'$(\times \ 1^-3\)\rightarrow$',fontsize=20)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		# n = 400
		# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
		# xticks = [int(k * n) for k in self.list_of_K]
		# plt.xticks(range(len(self.list_of_K)), xticks)
		# yticks = [-0.5,0.0,0.5,1.0,1.5]
		# plt.yticks(yticks)



		plt.savefig(image_file + '.pdf',dpi=600)
		plt.savefig(image_file + '.png', dpi=600)
		plt.close()


def main():
	option = sys.argv[1]   #sigm gauss
	file_name = ''
	list_of_std = []
	if option=='sigm':
		list_of_std=[0.001]
		file_name = 'sigmoid_n_240_d_1_inc_noise'
	else:
		list_of_std = [0.001]
		file_name = 'gauss_n_240_d_1_inc_noise'

	list_of_lamb=[0.001]
	list_of_K = [0.2,0.4,0.6,0.8]
	path = '../Synthetic_Fig3/'
	data_file = path + file_name
	obj=Synthetic_exp3(data_file, list_of_K, list_of_lamb, list_of_std )
	image_file_pre = path +file_name.split('_')[0]
	image_file_name =path+ 'demo_'+option+'_n_'
	obj.eval_loop( image_file_pre,image_file_name,option)
	
if __name__=="__main__":
	main()

