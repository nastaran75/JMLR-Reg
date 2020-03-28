import sys

import matplotlib
from brewer2mpl import brewer2mpl

from myutil import *
import numpy as np
import numpy.linalg as LA
import getopt


def parse_command_line_input(list_of_option, list_of_file_name):
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, 's:l:f:', ['std', 'lamb', 'file_name'])

    std = 0.001
    lamb = 0.001
    option = 'gauss'
    file_name = ''

    for opt, arg in opts:
        if opt == '-s':
            std = float(arg)

        if opt == '-l':
            lamb = float(arg)

        if opt == '-f':
            for file_name_i in list_of_file_name:
                if file_name_i.startswith(arg):
                    file_name = file_name_i

    return std, lamb, file_name


class plot_triage_real:
    def __init__(self, list_of_K, list_of_std, list_of_lamb, list_of_option, list_of_test_option, flag_synthetic=None):
        self.list_of_K = list_of_K
        self.list_of_std = list_of_std
        self.list_of_lamb = list_of_lamb
        self.list_of_option = list_of_option
        self.list_of_test_option = list_of_test_option
        self.flag_synthetic = flag_synthetic

    def get_avg_error_vary_K(self, res_file, image_path, test_method):

        res = load_data(res_file)
        for std in self.list_of_std:
            for lamb in self.list_of_lamb:
                plot_obj = {}
                plot_arr = np.zeros((len(self.list_of_option), len(self.list_of_K)))
                for option, ind in zip(self.list_of_option, range(len(self.list_of_option))):
                    err_K_tr = []
                    err_K_te = []
                    for K in self.list_of_K:
                        err_K_te.append(res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method][test_method]['error'])
                    plot_obj[option] = {'train': err_K_tr, 'test': err_K_te}
                    plot_arr[ind] = np.array(err_K_te)
                self.plot_err_vs_K(image_path, plot_obj, test_method)

    def get_avg_error_vary_testmethod(self, res_file, image_path,option):

        res = load_data(res_file)
        for std in self.list_of_std:
            for lamb in self.list_of_lamb:
                plot_obj = {}
                plot_arr = np.zeros((len(self.list_of_option), len(self.list_of_K)))
                for ind,test_method in enumerate(self.list_of_test_option):
                    err_K_tr = []
                    err_K_te = []
                    for K in self.list_of_K:
                        err_K_te.append(res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method][test_method]['error'])
                    plot_obj[test_method] = {'train': err_K_tr, 'test': err_K_te}
                    plot_arr[ind] = np.array(err_K_te)
                self.plot_err_vs_testmethod(image_path, plot_obj)

    def plot_err_vary_std_K(self, res_file):
        res = load_data(res_file)
        plot_arr = np.zeros((len(self.list_of_std), len(self.list_of_K)))
        for std, std_ind in zip(self.list_of_std, range(len(self.list_of_std))):
            for lamb in self.list_of_lamb:
                for test_method in self.list_of_test_option:
                    plot_obj = {}
                    for option in self.list_of_option:
                        if option != 'greedy':
                            continue
                        print option
                        # err_K_tr=[]
                        err_K_te = []
                        for K in self.list_of_K:
                            # err_K_tr.append(res[str(std)][str(K)][str(lamb)][option]['train_res']['error'])
                            err_K_te.append(res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method]['error'])
            plot_arr[std_ind] = np.array(err_K_te)
            plt.yticks([.001, .002, .003, .004])
            plt.plot(err_K_te, label=std)
        plt.legend()
        plt.show()

    def write_to_txt(self, res_arr, res_file_txt):
        with open(res_file_txt, 'w') as f:
            for row in res_arr:
                f.write('\t'.join(map(str, row)) + '\n')

    def plot_err_vs_K(self, image_file, plot_obj, test_method):
        matplotlib.rcParams['text.usetex'] = True
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=21)
        plt.rc('ytick', labelsize=21)
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.93)
        bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
        color_list = bmap.mpl_colors
        key = 'test'
        for idx,option in enumerate(plot_obj.keys()):
            err = [x*1000 for x in plot_obj[option][key]]
            if option=='kl_triage':
                label = 'Triage'
            if option == 'distort_greedy':
                label = 'Distorted greedy'
            if option=='greedy':
                label = 'Greedy'
            if option=='diff_submod':
                label='DS'
            if option=='RLSR_Reg':
                label = 'CRR'
            plt.plot(err, label=label, linewidth=3, marker='o',
                     markersize=10,color = color_list[idx])

        # plt.plot(plot_obj['greedy']['test'], label='Greedy', linewidth=3,  marker='o',
        #          markersize=10, color=color_list[0])
        # plt.plot(plot_obj['distort_greedy']['test'], label='Distorted greedy ', linewidth=3,
        #          marker='o', markersize=10, color=color_list[1])
        # plt.plot(plot_obj['diff_submod']['test'], label='DS', linewidth=3, marker='o',
        #          markersize=10, color=color_list[2])
        # plt.plot(plot_obj['kl_triage']['test'], label='Triage', linewidth=3,  marker='o',
        #          markersize=10, color=color_list[3])
        # plt.grid()
        plt.figtext(0.115, 0.95, r'$\times 10^{-3}$', fontsize=17)
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [4,0,1, 2,3]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 15}, frameon=False,
                   handlelength=0.2)
        plt.xlabel(r'$n/ | \mathcal{V} | $', fontsize=25)
        ax.set_ylabel(r'\textbf{MSE}', fontsize=25,
                      labelpad=8)  # ('MSE' +r'$(\times \ 1^-3\)\rightarrow$',fontsize=20)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # n = 400
        # xticks = [int(k * n) for k in self.list_of_K]
        plt.xticks(range(len(self.list_of_K)), self.list_of_K)
        plt.savefig(image_file + '.pdf', dpi=600)
        plt.savefig(image_file + '.png', dpi=600)
        # plt.show()
        save(plot_obj, image_file)
        plt.close()

    def plot_err_vs_testmethod(self, image_file, plot_obj):
        matplotlib.rcParams['text.usetex'] = True
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=21)
        plt.rc('ytick', labelsize=21)
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.93)
        bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
        color_list = bmap.mpl_colors
        key = 'test'
        for idx,option in enumerate(plot_obj.keys()):
            err = [x*1000 for x in plot_obj[option][key]]
            print option
            if option=='nearest':
                label = 'Nearest neighbor'
            if option == 'LR':
                label = 'Logistic regression'
            if option=='MLP':
                label = 'Multilayer perceptron'

            plt.plot(err, label=label, linewidth=3, marker='o',
                     markersize=10,color = color_list[idx])

        # plt.plot(plot_obj['greedy']['test'], label='Greedy', linewidth=3,  marker='o',
        #          markersize=10, color=color_list[0])
        # plt.plot(plot_obj['distort_greedy']['test'], label='Distorted greedy ', linewidth=3,
        #          marker='o', markersize=10, color=color_list[1])
        # plt.plot(plot_obj['diff_submod']['test'], label='DS', linewidth=3, marker='o',
        #          markersize=10, color=color_list[2])
        # plt.plot(plot_obj['kl_triage']['test'], label='Triage', linewidth=3,  marker='o',
        #          markersize=10, color=color_list[3])
        # plt.grid()
        plt.figtext(0.115, 0.95, r'$\times 10^{-3}$', fontsize=17)
        # handles, labels = plt.gca().get_legend_handles_labels()
        # order = [2, 3, 1, 0]
        plt.legend(prop={'size': 17}, frameon=False,
                   handlelength=0.2)
        plt.xlabel(r'$n/ | \mathcal{V} | $', fontsize=25)
        ax.set_ylabel(r'\textbf{MSE}', fontsize=25,
                      labelpad=4)  # ('MSE' +r'$(\times \ 1^-3\)\rightarrow$',fontsize=20)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # n = 400
        # xticks = [int(k * n) for k in self.list_of_K]
        plt.xticks(range(len(self.list_of_K)), self.list_of_K)
        plt.savefig(image_file + '.pdf', dpi=600)
        plt.savefig(image_file + '.png', dpi=600)
        # plt.show()
        save(plot_obj, image_file)
        plt.close()


    def get_nearest_human(self, dist, tr_human_ind):

        # start= time.time()
        n_tr = dist.shape[0]
        human_dist = float('inf')
        machine_dist = float('inf')
        for d, tr_ind in zip(dist, range(n_tr)):
            if tr_ind in tr_human_ind:
                if d < human_dist:
                    human_dist = d
            else:
                if d < machine_dist:
                    machine_dist = d
        # print 'Time required -----> ', time.time() - start , ' seconds'
        return (human_dist - machine_dist)

    def classification_get_test_error(self, res_obj, dist_mat,test_method, X_tr, x, y, y_h=None, c=None, K=None):

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
        if test_method=='MLP':
            model = MLPClassifier(max_iter=500)
        if test_method=='LR':
            model = LogisticRegression(max_iter=500,solver='liblinear')
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

    def get_test_error(self, res_obj, dist_mat, x, y, y_h=None, c=None, K=None):

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
        diff_arr = [self.get_nearest_human(dist, subset) for dist in dist_mat]
        # print 'Time required -----> ', time.time() - start , ' seconds'

        indices = np.argsort(np.array(diff_arr))
        subset_te_r = indices[:no_human]
        subset_machine_r = indices[no_human:]

        if subset_te_r.size == 0:
            error_r = err_m.sum() / float(n)
        else:
            error_r = (err_h[subset_te_r].sum() + err_m.sum() - err_m[subset_te_r].sum()) / float(n)

        subset_te_n = np.array([int(i) for i in range(len(diff_arr)) if diff_arr[i] < 0])
        # print 'subset size test', subset_te_n.shape
        subset_machine_n = np.array([int(i) for i in range(len(diff_arr)) if i not in subset_te_n])
        print 'sample to human--> ', str(subset_te_n.shape[0]), ', sample to machine--> ', str(
            subset_machine_n.shape[0])

        if subset_te_n.size == 0:
            error_n = err_m.sum() / float(n)
        else:
            error_n = (err_h[subset_te_n].sum() + err_m.sum() - err_m[subset_te_n].sum()) / float(n)

        error_n = {'error': error_n, 'human_ind': subset_te_n, 'machine_ind': subset_machine_n}
        error_r = {'error': error_r, 'human_ind': subset_te_r, 'machine_ind': subset_machine_r}
        return error_n, error_r

    def plot_test_allocation(self, train_obj, test_obj, plot_file_path):

        x = train_obj['human']['x']
        y = train_obj['human']['y']
        plt.scatter(x, y, c='blue', label='train human')

        x = train_obj['machine']['x']
        y = train_obj['machine']['y']
        plt.scatter(x, y, c='green', label='train machine')

        x = test_obj['machine']['x'][:, 0].flatten()
        y = test_obj['machine']['y']
        plt.scatter(x, y, c='yellow', label='test machine')

        x = test_obj['human']['x'][:, 0].flatten()
        y = test_obj['human']['y']
        plt.scatter(x, y, c='red', label='test human')

        plt.legend()
        plt.grid()
        plt.xlabel('<-----------x------------->')
        plt.ylabel('<-----------y------------->')
        plt.close()

    def get_train_error(self, plt_obj, x, y, y_h=None, c=None):
        subset = plt_obj['subset']

        w = plt_obj['w']
        n = y.shape[0]
        if y_h == None:
            err_h = c
        else:
            err_h = (y_h - y) ** 2

        y_m = x.dot(w)
        err_m = (y_m - y) ** 2
        error = (err_h[subset].sum() + err_m.sum() - err_m[subset].sum()) / float(n)
        return {'error': error}

    def compute_result(self, res_file, data_file, option, test_method, image_file_prefix):
        data = load_data(data_file)
        res = load_data(res_file)
        print option
        for std, i0 in zip(self.list_of_std, range(len(self.list_of_std))):
            for K, i1 in zip(self.list_of_K, range(len(self.list_of_K))):
                for lamb, i2 in zip(self.list_of_lamb, range(len(self.list_of_lamb))):
                    print res[str(std)][str(K)][str(lamb)].keys()
                    res_obj = res[str(std)][str(K)][str(lamb)][option]
                    suffix = '_' + option + '_std_' + str(std) + '_K_' + str(K) + '_lamb_' + str(lamb)
                    image_file = image_file_prefix + suffix  # '../Synthetic_data/demo/'
                    self.plot_subset_allocation(data['X'], data['Y'], res_obj['w'], res_obj['subset'], image_file)
                    print 'std', str(std), '  K', str(K), '  lamb  ', str(lamb)
                    train_res = self.get_train_error(res_obj, data['X'], data['Y'], y_h=None, c=data['c'][str(std)])

                    if 'test_res' not in res[str(std)][str(K)][str(lamb)][option]:
                        res[str(std)][str(K)][str(lamb)][option]['test_res'] = {}
                    if test_method == 'nearest':
                        test_res_n, test_res_r = self.get_test_error(res_obj, data['dist_mat'], data['test']['X'],
                                                                 data['test']['Y'], y_h=None,
                                                                 c=data['test']['c'][str(std)], K=K)

                    else:
                        test_res_n, test_res_r = self.classification_get_test_error(res_obj, data['dist_mat'],test_method,
                                                                                    data['X'], data['test']['X'],
                                                                                    data['test']['Y'], y_h=None,
                                                                                  c=data['test']['c'][str(std)], K=K)

                    if 'test_res' not in res[str(std)][str(K)][str(lamb)][option]:
                        res[str(std)][str(K)][str(lamb)][option]['test_res'] = {}
                    if 'train_res' not in res[str(std)][str(K)][str(lamb)][option]:
                        res[str(std)][str(K)][str(lamb)][option]['train_res'] = {}

                    res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method] = {'ranking': test_res_r,
                                                                            test_method: test_res_n}
                    res[str(std)][str(K)][str(lamb)][option]['train_res'][test_method] = train_res
        save(res, res_file)

    def plot_subset_allocation(self, X, Y, w, subset, image_file):

        x = X[:, 0].flatten()[subset]
        y = Y[subset]
        plt.scatter(x, y, c='blue', label='human')

        subset_c = np.array([i for i in range(Y.shape[0]) if i not in subset])
        x = X[:, 0].flatten()[subset_c]
        y = Y[subset_c]
        plt.scatter(x, y, c='green', label='machine')

        x = X[:, 0].flatten()
        y = X.dot(w)
        plt.scatter(x, y, c='yellow', label='prediction')

        plt.legend()
        plt.grid()
        plt.xlabel('<-----------x------------->')
        plt.ylabel('<-----------y------------->')
        plt.close()

    # plt.show()

    def merge_results(self, input_res_files, merged_res_file):

        res = {}
        for std in self.list_of_std:
            if str(std) not in res:
                res[str(std)] = {}
            for K in self.list_of_K:
                if str(K) not in res[str(std)]:
                    res[str(std)][str(K)] = {}
                for lamb in self.list_of_lamb:
                    if str(lamb) not in res[str(std)][str(K)]:
                        res[str(std)][str(K)][str(lamb)] = {}
                    r = load_data(input_res_files[str(lamb)])
                    res[str(std)][str(K)][str(lamb)] = r[str(std)][str(K)][str(lamb)]
        save(res, merged_res_file)

    def split_res_over_K(self, data_file, res_file, unified_K, option):
        res = load_data(res_file)
        for std in self.list_of_std:
            if str(std) not in res:
                res[str(std)] = {}
            for K in self.list_of_K:
                if str(K) not in res[str(std)]:
                    res[str(std)][str(K)] = {}
                for lamb in self.list_of_lamb:
                    if str(lamb) not in res[str(std)][str(K)]:
                        res[str(std)][str(K)][str(lamb)] = {}

                    if option not in res[str(std)][str(K)][str(lamb)]:
                        res[str(std)][str(K)][str(lamb)][option] = {}
                        # if 'test_res' not in res[str(std)][str(K)][str(lamb)][option]:
                        #     res[str(std)][str(K)][str(lamb)][option]['test_res'] = {}
                        #     for test_method in self.list_of_test_option:
                        #         if test_method not in res[str(std)][str(K)][str(lamb)][option]['test_res']:
                        #             res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method] = {}


                        if K != unified_K:
                            res_dict = res[str(std)][str(unified_K)][str(lamb)][option]
                            if res_dict:
                                res[str(std)][str(K)][str(lamb)][option] = self.get_res_for_subset(data_file, res_dict,
                                                                                               lamb, K)
        save(res, res_file)

    def get_optimal_pred(self, data, subset, lamb):
        n, dim = data['X'].shape
        subset_c = np.array([int(i) for i in range(n) if i not in subset])
        X_sub = data['X'][subset_c].T
        Y_sub = data['Y'][subset_c]
        subset_c_l = n - subset.shape[0]
        return LA.inv(lamb * subset_c_l * np.eye(dim) + X_sub.dot(X_sub.T)).dot(X_sub.dot(Y_sub))

    def get_res_for_subset(self, data_file, res_dict, lamb, K):
        data = load_data(data_file)
        curr_n = int(data['X'].shape[0] * K)
        subset_tr = res_dict['subset'][:curr_n]
        w = self.get_optimal_pred(data, subset_tr, lamb)
        return {'w': w, 'subset': subset_tr}


def main():
    list_of_option = ['greedy', 'RLSR_Reg','distort_greedy', 'kl_triage', 'diff_submod']
    list_of_file_name = ['sigmoid', 'gauss']

    # specify std, lamb and file_name as specified in ReadMe.txt
    std, lamb, file_name = parse_command_line_input(list_of_option, list_of_file_name)
    print file_name

    list_of_std = [std]
    list_of_lamb = [lamb]
    list_of_K = [0.1, 0.2, 0.3, .4, 0.5, .6, .7, .8, .9]

    list_of_test_option = ['nearest','LR','MLP']
    path = '../Synthetic_data/'
    data_file = path + 'data_dict_' + file_name
    res_file = path + 'res_' + file_name #+ str(std) + '_' + str(lamb)
    image_path = path + 'plot_vary_K_' + file_name
    image_demo_prefix = path + 'demo_' + file_name

    obj = plot_triage_real(list_of_K, list_of_std, list_of_lamb, list_of_option, list_of_test_option)
    if 'sigmoid' in file_name:
        prefix = 'sigm'
    if 'gauss' in file_name:
        prefix = 'gauss'

    for test_method in list_of_test_option:
        for option in list_of_option:
            if option not in ['diff_submod','RLSR_Reg']:
                obj.split_res_over_K(data_file, res_file, 0.99, option)
            obj.compute_result(res_file, data_file, option, test_method, image_demo_prefix)
        suffix = '_'+ test_method + '_classifier'
        if test_method == 'nearest':
            suffix = ''
        new_image_path = path + prefix+'_k_baseline_new' + suffix # + str(std) + '_' + str(lamb)
        obj.get_avg_error_vary_K(res_file, new_image_path, test_method)  # produces figure 2 of the paper
    varytestpath = path + prefix + '_k_baseline_vary_testmethod'# + str(std) + '_' + str(lamb)
    obj.get_avg_error_vary_testmethod(res_file, varytestpath,'greedy')


if __name__ == "__main__":
    main()
