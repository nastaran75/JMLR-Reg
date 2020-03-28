import time
import sys
import os
from myutil import *
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
from generate_data import generate_data
from matplotlib import rc
import matplotlib
import brewer2mpl

class plot_triage_real:
	def __init__(self,list_of_K,list_of_std, list_of_lamb,list_of_option,list_of_test_option, flag_synthetic=None):
		self.list_of_K=list_of_K
		self.list_of_std=list_of_std
		self.list_of_lamb=list_of_lamb
		self.list_of_option=list_of_option
		self.list_of_test_option = list_of_test_option
		self.flag_synthetic = flag_synthetic

	def get_avg_error_vary_K(self,res_file,data_file,test_method,dataset,path,image_path):

		def smooth( vec ):
			vec = np.array(vec)
			tmp = np.array( [ (.25*vec[ind]+.5*vec[ind-1]+.25*vec[ind+1]) for ind  in range(1,vec.shape[0]-1)  ])
			vec[0] = vec[0]*.75+vec[1]*.25
			vec[-1] = vec[-1]*.75+vec[-2]*.25
			vec[1:-1] = tmp
			return vec
		
		res=load_data(res_file)
		data = load_data(data_file)
		if dataset=='stare5':
			dataset = 'stareD'
		if dataset=='stare11':
			dataset = 'stareH'
		if dataset == 'sigmoid':
			dataset = 'sigm'
		# n = data['X'].shape[0]  #for synthetic
		real = ['messidor','stareD','stareH']
		synthetic = ['sigm','gauss']
		if dataset in real:
			multtext = r'$\times 10^{-1}$'
			labeltext = r'$\rho_c = $'
			if test_method =='MLP' or test_method=='LR':
				savepathpdf = path+dataset+'U_new_'+test_method+'_classifier.pdf'
				savepathpng = path+dataset + 'U_new_' + test_method + '_classifier.png'
			if test_method == 'nearest':
				savepathpdf = path+dataset + 'U_new.pdf'
				savepathpng = path+dataset + 'U_new.png'
		if dataset in synthetic:
			multtext = r'$\times 10^{-3}$'
			labeltext = r'$\sigma_2 = $'
			if test_method=='MLP' or test_method == 'LR':
				suffix = '_' + test_method + '_classifier'
			if test_method == 'nearest':
				suffix = ''
			savepathpdf = path+dataset+'_k_human_new'+suffix+'.pdf'
			savepathpng = path + dataset + '_k_human_new' + suffix + '.png'


		matplotlib.rcParams['text.usetex'] = True
		plt.rc('font', family='serif')
		plt.rc('xtick', labelsize=21)
		plt.rc('ytick', labelsize=21)
		fig, ax = plt.subplots()
		fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.93)
		bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
		color_list = bmap.mpl_colors
		plt.figtext(0.115, 0.96, multtext, fontsize=17)
		# color_list = ['mediumaquamarine','violet','cornflowerblue','coral','mediumpurple']
		for idx,std in enumerate(self.list_of_std):
			for lamb in self.list_of_lamb:
				# for test_method in self.list_of_test_option:
					for option,ind in zip(self.list_of_option, range( len(self.list_of_option)) ):
						option_flag=0
						err_K_tr=[]
						err_K_te=[]
						for K in self.list_of_K:
							err_K_tr.append(res[str(std)][str(K)][str(lamb)][option]['train_res']['error'])
							err_K_te.append(res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method][test_method]['error'])
						err_K_te_mult = [10*err for err in err_K_te]  # for synthetic
						ax.plot((err_K_te_mult), label=labeltext +str(std), linewidth=3, marker='o',
                     markersize=10,color=color_list[idx])
		# plt.grid()

		ax.legend(prop={'size': 18},frameon=False, handlelength=0.2,loc='best')
		plt.xlabel(r'$n/ | \mathcal{V} | $', fontsize=25)
		ax.set_ylabel(r'\textbf{MSE}',fontsize=25,labelpad=3)#('MSE' +r'$(\times \ 1^-3\)\rightarrow$',fontsize=20)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		# xticks = [int(k * n) for k in self.list_of_K]
		plt.xticks(range(len(self.list_of_K)), self.list_of_K)
		# plt.savefig(res_file+'_'+test_method+'_plot.pdf')
		# plt.savefig(res_file +'_'+test_method+ '_plot.png')


		plt.savefig(savepathpdf)
		plt.savefig(savepathpng)
		plt.close()

	def get_avg_error_vary_testmethod(self, res_file, data_file, dataset, path,std):

		res = load_data(res_file)
		data = load_data(data_file)
		if dataset == 'stare5':
			dataset = 'stareD'
		if dataset == 'stare11':
			dataset = 'stareH'
		if dataset == 'sigmoid':
			dataset = 'sigm'
		# n = data['X'].shape[0]  #for synthetic
		real = ['messidor', 'stareD', 'stareH']
		synthetic = ['sigm', 'gauss']
		if dataset in real:
			savepathpdf = path + dataset + 'U_' + str(std)+ '_vary_testmethod.pdf'
			savepathpng = path + dataset + 'U_' + str(std) + '_vary_testmethod.png'
			multtext = r'$\times 10^{-1}$'

		if dataset in synthetic:
			savepathpdf = path + dataset + '_k_human_' + str(std) + '_vary_testmethod.pdf'
			savepathpng = path + dataset + '_k_human_' + str(std) + '_vary_testmethod.png'
			multtext = r'$\times 10^{-3}$'

		matplotlib.rcParams['text.usetex'] = True
		plt.rc('font', family='serif')
		plt.rc('xtick', labelsize=21)
		plt.rc('ytick', labelsize=21)
		fig, ax = plt.subplots()
		fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.93)
		bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
		color_list = bmap.mpl_colors
		plt.figtext(0.115, 0.96, multtext, fontsize=17)
		# color_list = ['mediumaquamarine','violet','cornflowerblue','coral','mediumpurple']
		for lamb in self.list_of_lamb:
			# for test_method in self.list_of_test_option:
			for option, ind in zip(self.list_of_option, range(len(self.list_of_option))):
				option_flag = 0

				for idx,test_method in enumerate(self.list_of_test_option):
					if test_method=='nearest':
						labeltext = 'Nearest neighbor'
					if test_method == 'MLP':
						labeltext = 'Multilayer perceptron'
					if test_method == 'LR':
						labeltext = 'Logistic regression'
					err_K_tr = []
					err_K_te = []
					for K in self.list_of_K:
						err_K_tr.append(res[str(std)][str(K)][str(lamb)][option]['train_res']['error'])
						err_K_te.append(
							res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method][test_method]['error'])
					err_K_te_mult = [10 * err for err in err_K_te]  # for synthetic
					ax.plot((err_K_te_mult), label=labeltext, linewidth=3, marker='o',
							markersize=10, color=color_list[idx])
		# plt.grid()

		ax.legend(prop={'size': 18}, frameon=False, handlelength=0.2, loc='best')
		plt.xlabel(r'$n/ | \mathcal{V} | $', fontsize=25)
		ax.set_ylabel(r'\textbf{MSE}', fontsize=25,
					  labelpad=3)  # ('MSE' +r'$(\times \ 1^-3\)\rightarrow$',fontsize=20)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		if dataset == 'sigm':
			ax.set_ylim([-0.2, 12.5])
		if dataset == 'gauss' or dataset=='stareH':
			ax.set_ylim([0, 2.5])
		if dataset == 'stareD':
			ax.set_ylim([0,1])
		if dataset == 'messidor':
			ax.set_ylim([0,4])
		# xticks = [int(k * n) for k in self.list_of_K]
		plt.xticks(range(len(self.list_of_K)), self.list_of_K)
		# plt.savefig(res_file+'_'+test_method+'_plot.pdf')
		# plt.savefig(res_file +'_'+test_method+ '_plot.png')

		plt.savefig(savepathpdf)
		plt.savefig(savepathpng)
		plt.close()

	def plot_err_vary_std_K(self, res_file, res_file_txt, n):
		res=load_data(res_file)
		plot_arr= np.zeros( (len(self.list_of_std), len(self.list_of_K)))
		K_axis = np.array([ int(k*n*0.8) for k in self.list_of_K ])
		for std,std_ind in zip(self.list_of_std, range( len(self.list_of_std) )):
			for lamb in self.list_of_lamb:
				for test_method in self.list_of_test_option:
					# suffix='lamb_'+str(lamb)+'_'+test_method
					# image_file=image_path+suffix.replace('.','_')
					plot_obj={}
					for option in self.list_of_option:
						# err_K_tr=[]
						err_K_te=[]
						for K in self.list_of_K:
							# err_K_tr.append(res[str(std)][str(K)][str(lamb)][option]['train_res']['error'])
							err_K_te.append(res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method]['error'])
			plot_arr[std_ind]=np.array( err_K_te )
			plt.plot( err_K_te, label = str(std),linewidth=8,linestyle='--',marker='o', markersize=10)
		plt.grid()
		plt.xticks(range(len(self.list_of_K)),K_axis)
		# plt.legend()
		plt.xlabel('K')
		plt.ylabel('Average Squared Error')
		plt.title('Average Squared Error  Vs Deviation of human error')
		# plt.savefig(res_file_txt+'.jpg')
		plt.savefig(res_file_txt+'.pdf')
		plt.show()
		# K = np.array([ int(k*n*0.8) for k in self.list_of_K ])
		
		plot_arr = np.hstack(( K_axis.reshape(K_axis.shape[0],1), plot_arr.T))
		self.write_to_txt(plot_arr, res_file_txt)

	def write_to_txt(self, res_arr, res_file_txt):
		with open( res_file_txt, 'w') as f:
			for row in res_arr:
				f.write( '\t'.join(map( str, row)) + '\n' )

	def plot_err_vs_K(self,image_file,plot_obj):
		for option in plot_obj.keys():
			plt.plot( plot_obj[option]['test'], label=option, linewidth=8,linestyle='--',marker='o', markersize=10)
		# plt.plot(plot_obj['greedy']['test'],label='GR',linewidth=8,linestyle='--',marker='o', markersize=10,color='red')
		# plt.plot(plot_obj['diff_submod']['test'],label='DS',linewidth=8,linestyle='-',marker='o', markersize=10,color='blue')
		# plt.plot(plot_obj['distort_greedy']['test'],label='DG',linewidth=8,linestyle='--',marker='o', markersize=10,color='green')
		# plt.plot(plot_obj['stochastic_distort_greedy']['test'],label='SDG',linewidth=8,linestyle='-',marker='o', markersize=10,color='yellow')
		# plt.plot(plot_obj['kl_triage']['test'],label='KL',linewidth=8,linestyle='-',marker='o', markersize=10,color='black')
		plt.grid()
		plt.legend()
		plt.xlabel('K')
		plt.ylabel('Average Squared Error')
		plt.title('Average Squared Error')
		plt.xticks(range(len(self.list_of_K)),self.list_of_K)
		plt.savefig(image_file+'.pdf',dpi=600, bbox_inches='tight')
		plt.savefig(image_file+'.jpg',dpi=600, bbox_inches='tight')
		# plt.show()
		save(plot_obj,image_file)
		plt.close()

	def get_nearest_human(self,dist,tr_human_ind):
		
		# start= time.time()
		n_tr=dist.shape[0]
		human_dist=float('inf')
		machine_dist=float('inf')
		for d,tr_ind in zip(dist,range(n_tr)):
			if tr_ind in tr_human_ind:
				if d < human_dist:
					human_dist=d
			else:
				if d < machine_dist:
					machine_dist=d
		# print 'Time required -----> ', time.time() - start , ' seconds'
		return (human_dist -machine_dist)

	def classification_get_test_error(self, res_obj, dist_mat,mode, X_tr, x, y, y_h=None, c=None, K=None):

		w = res_obj['w']
		subset = res_obj['subset']
		n, tr_n = dist_mat.shape
		no_human = int((subset.shape[0] * n) / float(tr_n))

		y_m = x.dot(w)
		err_m = (y - y_m) ** 2
		if y_h == None:
			err_h = c
		else:
			err_h = (y - y_h) ** 2

		# start = time.time()
		# diff_arr = [self.get_nearest_human(dist, subset) for dist in dist_mat]
		# print 'Time required -----> ', time.time() - start , ' seconds'

		# indices = np.argsort(np.array(diff_arr))
		# subset_te_r = indices[:no_human]
		# subset_machine_r = indices[no_human:]

		y_tr = np.zeros(tr_n, dtype='uint')
		y_tr[subset] = 1  # human label = 1
		from sklearn.neural_network import MLPClassifier
		from sklearn.linear_model import LogisticRegression
		if mode == 'MLP':
			model = MLPClassifier(max_iter=500)
		if mode == 'LR':  #logistic regression
			model = LogisticRegression(solver='liblinear')
		model.fit(X_tr, y_tr)
		y_pred = model.predict(x)


		subset_te_r = []
		subset_machine_r = []
		for idx, label in enumerate(y_pred):
			if label == 1:
				subset_te_r.append(idx)
			else:
				subset_machine_r.append(idx)

		subset_machine_r = np.array(subset_machine_r)
		subset_te_r = np.array(subset_te_r)

		if subset_te_r.size == 0:
			error_r = err_m.sum() / float(n)
		else:
			error_r = (err_h[subset_te_r].sum() + err_m.sum() - err_m[subset_te_r].sum()) / float(n)

		subset_te_n = np.array([int(i) for i in range(len(y_pred)) if y_pred[i] == 1])
		# print 'subset size test', subset_te_n.shape
		subset_machine_n = np.array([int(i) for i in range(len(y_pred)) if i not in subset_te_n])
		# print 'sample to human--> ' , str(subset_te_n.shape[0]), ', sample to machine--> ', str( subset_machine_n.shape[0])

		if subset_te_n.size == 0:
			error_n = err_m.sum() / float(n)
		else:
			error_n = (err_h[subset_te_n].sum() + err_m.sum() - err_m[subset_te_n].sum()) / float(n)

		# return {'error':error, 'human_ind':subset_te, 'machine_ind':subset_machine}
		error_n = {'error': error_n, 'human_ind': subset_te_n, 'machine_ind': subset_machine_n}
		error_r = {'error': error_r, 'human_ind': subset_te_r, 'machine_ind': subset_machine_r}
		return error_n, error_r

	def get_test_error(self,res_obj,dist_mat,x,y,y_h=None,c=None,K=None):
		
		w=res_obj['w']
		subset=res_obj['subset']
		n,tr_n=dist_mat.shape
		no_human=int((subset.shape[0]*n)/float(tr_n))

		y_m=x.dot(w)
		err_m=(y-y_m)**2
		if y_h==None:
			err_h=c  
		else: 
			err_h=(y-y_h)**2

		# start = time.time()
		diff_arr=[ self.get_nearest_human(dist,subset) for dist in dist_mat]
		# print 'Time required -----> ', time.time() - start , ' seconds'

		indices=np.argsort(np.array(diff_arr))
		subset_te_r = indices[:no_human]
		subset_machine_r=indices[no_human:]

		if subset_te_r.size==0:
			error_r =  err_m.sum()/float(n)
		else:
			error_r = ( err_h[subset_te_r].sum() + err_m.sum() - err_m[subset_te_r].sum() ) /float(n)


		subset_te_n = np.array([int(i)  for i in range(len(diff_arr)) if diff_arr[i] < 0 ])
		# print 'subset size test', subset_te_n.shape
		subset_machine_n = np.array([int(i)  for i in range(len(diff_arr)) if i not in subset_te_n ])
		print 'sample to human--> ' , str(subset_te_n.shape[0]), ', sample to machine--> ', str( subset_machine_n.shape[0])

		if subset_te_n.size==0:
			error_n =  err_m.sum()/float(n)
		else:
			error_n = ( err_h[subset_te_n].sum() + err_m.sum() - err_m[subset_te_n].sum() ) /float(n)

		# return {'error':error, 'human_ind':subset_te, 'machine_ind':subset_machine}
		error_n={'error':error_n, 'human_ind':subset_te_n, 'machine_ind':subset_machine_n}
		error_r={'error':error_r, 'human_ind':subset_te_r, 'machine_ind':subset_machine_r}
		return error_n, error_r

	def plot_test_allocation(self,train_obj,test_obj,plot_file_path):

		x=train_obj['human']['x']
		y=train_obj['human']['y']
		plt.scatter(x,y,c='blue',label='train human')

		x=train_obj['machine']['x']
		y=train_obj['machine']['y']
		plt.scatter(x,y,c='green',label='train machine')

		x=test_obj['machine']['x'][:,0].flatten()
		y=test_obj['machine']['y']
		plt.scatter(x,y,c='yellow',label='test machine')

		x=test_obj['human']['x'][:,0].flatten()
		y=test_obj['human']['y']
		plt.scatter(x,y,c='red',label='test human')

		plt.legend()
		plt.grid()
		plt.xlabel('<-----------x------------->')
		plt.ylabel('<-----------y------------->')
		plt.savefig(plot_file_path,dpi=600, bbox_inches='tight')
		plt.close()

		# plt.show()
					
	def get_train_error(self,plt_obj,x,y,y_h=None,c=None):
		subset = plt_obj['subset']
		# print np.min(subset)
		# print np.max(subset)

		w=plt_obj['w']
		n=y.shape[0]
		if y_h==None:
			err_h=c
		else:
			err_h=(y_h-y)**2

		# print x.shape

		y_m= x.dot(w)
		err_m=(y_m-y)**2
		# print '-----------'
		# print err_h.shape
		# print err_m.shape
		# print '-----------'
		error = ( err_h[subset].sum()+err_m.sum() - err_m[subset].sum() ) /float(n)
		return {'error':error}

	def compute_result(self,res_file,data_file,option,test_method, image_file_prefix =None):
		data=load_data(data_file)
		res=load_data(res_file)
		for std,i0 in zip(self.list_of_std,range( len(self.list_of_std) )):
			for K,i1 in zip(self.list_of_K,range(len(self.list_of_K))):
				for lamb,i2 in zip(self.list_of_lamb,range(len(self.list_of_lamb))):
					res_obj=res[str(std)][str(K)][str(lamb)][option]
					train_res = self.get_train_error(res_obj,data['X'],data['Y'],y_h=None,c=data['c'][str(std)])
					if test_method=='nearest':
						test_res_n,test_res_r = self.get_test_error(res_obj,data['dist_mat'],data['test']['X'],data['test']['Y'],y_h=None,c=data['test']['c'][str(std)],K=K)
					else:
						test_res_n, test_res_r = self.classification_get_test_error(res_obj, data['dist_mat'],test_method,
																					data['X'], data['test']['X'],
																					data['test']['Y'], y_h=None,
																					c=data['test']['c'][str(std)], K=K)
					if 'test_res' not in res[str(std)][str(K)][str(lamb)][option]:
						res[str(std)][str(K)][str(lamb)][option]['test_res'] = {}
					res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method]={'ranking':test_res_r,test_method:test_res_n}
					res[str(std)][str(K)][str(lamb)][option]['train_res']=train_res
		save(res,res_file)

	def plot_subset_allocation( self, X, Y, w, subset, image_file):

		x=X[:,0].flatten()[subset]
		y=Y[subset]
		plt.scatter(x,y,c='blue',label='human')

		subset_c = np.array([ i for i in range( Y.shape[0]) if i not in  subset])
		x=X[:,0].flatten()[subset_c]
		y=Y[subset_c]
		plt.scatter(x,y,c='green',label='machine')

		
		x=X[:,0].flatten()
		y=X.dot(w)
		plt.scatter(x,y,c='yellow',label='prediction')

		plt.legend()
		plt.grid()
		plt.xlabel('<-----------x------------->')
		plt.ylabel('<-----------y------------->')
		plt.savefig(image_file+'.pdf',dpi=600, bbox_inches='tight')
		# plt.savefig(image_file+'.jpg',dpi=600, bbox_inches='tight')
		# plt.show()
		plt.close()
		# plt.show()
			
	def merge_results(self,input_res_files,merged_res_file):

		res={}
		for std in self.list_of_std:
			if str(std) not in res:
				res[str(std)]={}
			for K in self.list_of_K:
				if str(K) not in res[str(std)]:
					res[str(std)][str(K)]={}
				for lamb in self.list_of_lamb:
					if str(lamb) not in res[str(std)][str(K)]:
						res[str(std)][str(K)][str(lamb)]={}
					r=load_data(input_res_files[str(lamb)])
					# print r['0.0'].keys()
					# print res['0.0'].keys()
					res[str(std)][str(K)][str(lamb)] = r[str(std)][str(K)][str(lamb)]
		save(res,merged_res_file)

	def split_res_over_K(self,data_file,res_file,unified_K,option):
		res=load_data(res_file)
		for std in self.list_of_std:
			if str(std) not in res:
				res[str(std)]={}
			for K in self.list_of_K:
				if str(K) not in res[str(std)]:
					res[str(std)][str(K)]={}
				for lamb in self.list_of_lamb:
					if str(lamb) not in res[str(std)][str(K)]:
						res[str(std)][str(K)][str(lamb)]={}
					
					if option not in res[str(std)][str(K)][str(lamb)]:
						res[str(std)][str(K)][str(lamb)][option]={}

						if 'test_res' not in res[str(std)][str(K)][str(lamb)][option]:
							res[str(std)][str(K)][str(lamb)][option]['test_res'] = {}
							for test_method in self.list_of_test_option:
								if test_method not in res[str(std)][str(K)][str(lamb)][option]['test_res']:
									res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method] = {}

						if K != unified_K:
							res_dict = res[str(std)][str(unified_K)][str(lamb)][option]
							if res_dict:
								res[str(std)][str(K)][str(lamb)][option] = self.get_res_for_subset(data_file,res_dict,lamb,K)
		save(res,res_file)

	def get_optimal_pred(self,data,subset,lamb):
		
		n,dim= data['X'].shape
		subset_c=  np.array([int(i) for i in range(n) if i not in subset])	
		X_sub=data['X'][subset_c].T
		Y_sub=data['Y'][subset_c]
		subset_c_l=n-subset.shape[0]
		return LA.inv( lamb*subset_c_l*np.eye(dim) + X_sub.dot(X_sub.T) ).dot(X_sub.dot(Y_sub))

	def get_res_for_subset(self,data_file,res_dict,lamb,K):
		data=load_data(data_file)
		curr_n = int( data['X'].shape[0] * K )
		subset_tr = res_dict['subset'][:curr_n]
		w= self.get_optimal_pred(data,subset_tr,lamb)
		return {'w':w,'subset':subset_tr}

def main():
	file_name =sys.argv[1] #messidor,stare5,stare11,gauss,sigmoid
	if 'sigmoid'in file_name or 'gauss' in file_name:
		list_of_std = [0.01,0.02,0.03,0.04,0.05]
		path = '../Synthetic_data/'
		data_file = '../Synthetic_data/data_dict_'+file_name+'_fig_2_n500d5'
		res_file = '../Synthetic_data/res_'+file_name+'_fig_2_n500d5'
	else:
		list_of_std = [.2,.4,.6,.8]#
		path = '../Real_Data_Results/'#
		data_file = path + 'data/' + file_name+'_pca50'
		res_file = path + file_name + '_res_pca50'#
	# list_of_std=[0.01,0.02,0.03,0.04,0.05]#
	list_of_K = [ 0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 ] 
	list_of_option =['greedy']
	list_of_test_option = ['nearest','MLP','LR']#,'classification']
	#,'stare11','mesidor_re','messidor_rg','eyepac','chexpert7','chexpert8']
	if 'stare5' in file_name:
		list_of_lamb = [1]#[0.01]  # [0.001]#
	elif 'stare11' in file_name:
		list_of_lamb = [1]#[0.1]
	elif 'messidor' in file_name:
		list_of_lamb = [1]#[0.1]
	elif 'sigmoid' in file_name:
		list_of_lamb = [0.001]   #[0.001] for sig [0.005] for gauss
	else:
		list_of_lamb = [0.005]
	# path = '../Synthetic_data/'#'../Real_Data_Results/'#
	obj=plot_triage_real(list_of_K, list_of_std, list_of_lamb, list_of_option, list_of_test_option)

	for test_method in list_of_test_option:
		# data_file = '../Synthetic_data/data_dict_sigmoid_fig_2_n500d5' #path + 'data/' + file_name+'_pca50'#path + 'data/' + file_name+'_pca50'
		# res_file= '../Synthetic_data/res_sigmoid_fig_2_n500d5'#path + file_name + '_res_pca50'#
		# print('-'*50+'\n'+file_name+'\n\n'+'-'*50)
		for option in list_of_option:
			if True: # option not in [ 'diff_submod', 'stochastic_distort_greedy']:
				print('*'*10+'\n'+option+'\n'+'*'*10)
				unified_K = 0.99
				obj.split_res_over_K(data_file,res_file,unified_K,option)
				obj.compute_result(res_file,data_file,option,test_method, 'dummy')
		image_path = ''#path + 'Fig1/'+file_name+'_'
		obj.get_avg_error_vary_K(res_file,data_file,test_method,file_name,path,image_path)
	for std in list_of_std:
		obj.get_avg_error_vary_testmethod(res_file,data_file,file_name,path,std)
	

if __name__=="__main__":
	main()

