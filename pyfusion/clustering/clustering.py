'''
Shaun Haskey : 2May2013

Many routines for clustering and plotting.

'''
import numpy as np
import matplotlib.pyplot as pt
import math, time, copy, itertools, multiprocessing, os
from sklearn import mixture
from scipy.cluster import vq
from scipy.stats.distributions import vonmises
from scipy.stats.distributions import norm
import cPickle as pickle
import scipy.special as spec
import scipy.optimize as opt

def compare_several_kappa_values(clusters, pub_fig = 0, alpha = 0.05,decimation=10, labels=None, plot_style_list=None, filename='extraction_comparison.pdf',xaxis='sigma_eq',max_cutoff_value = 35, vline = None):
    '''xaxis can be sigma_eq, kappa_bar or sigma_bar
    '''
    fig, ax = pt.subplots()
    if pub_fig:
        cm_to_inch=0.393701
        import matplotlib as mpl
        old_rc_Params = mpl.rcParams
        mpl.rcParams['font.size']=8.0
        mpl.rcParams['axes.titlesize']=8.0#'medium'
        mpl.rcParams['xtick.labelsize']=8.0
        mpl.rcParams['ytick.labelsize']=8.0
        mpl.rcParams['lines.markersize']=5.0
        mpl.rcParams['savefig.dpi']=300
        fig.set_figwidth(8.48*cm_to_inch)
        fig.set_figheight(8.48*0.8*cm_to_inch)

    for cur_cluster,cur_label,plot_style in zip(clusters, labels,plot_style_list):
        std_bar, std_eq = sigma_eq_sigma_bar(cur_cluster.cluster_details["EM_VMM_kappas"], deg=True)
        averages = np.average(cur_cluster.cluster_details["EM_VMM_kappas"],axis=1)
        items = []
        kappa_cutoff_list = range(max_cutoff_value)
        for kappa_cutoff_tmp in kappa_cutoff_list:
            if xaxis=='sigma_eq':
                cluster_list = np.arange(len(averages))[std_eq<kappa_cutoff_tmp]
            elif xaxis=='sigma_bar':
                cluster_list = np.arange(len(averages))[std_bar<kappa_cutoff_tmp]
            else:
                cluster_list = np.arange(len(averages))[averages>kappa_cutoff_tmp]
            total = 0
            for i in cluster_list: total+= np.sum(cur_cluster.cluster_assignments==i)
            items.append(total)
        ax.plot(kappa_cutoff_list, items,plot_style,label=cur_label)
    ax.legend(loc='best',prop={'size':8.0})
    ax.set_ylabel('Number of Features')
    if vline!=None:
        ax.vlines(vline,ax.get_ylim()[0], ax.get_ylim()[1])
    if xaxis=='sigma_eq':
        ax.set_xlabel(r'$\sigma_{eq}$ cutoff')
    elif xaxis=='sigma_bar':
        ax.set_xlabel(r'$\bar{\sigma}$ cutoff')
    else:
        ax.set_xlabel(r'$\kappa$ cutoff')
    #fig.savefig('hello.eps', bbox_inches='tight', pad_inches=0)
    fig.savefig(filename, bbox_inches='tight', pad_inches=0.01)
    fig.canvas.draw(); fig.show()

def sigma_eq_sigma_bar(kappas, deg=False):
    std_circ = convert_kappa_std(kappas, deg=False)
    if len(kappas.shape)==2:
        std_bar = np.mean(std_circ,axis=1)
        std_eq = (np.product(std_circ,axis=1))**(1./kappas.shape[1])
    elif len(kappas.shape)==1:
        std_bar = np.mean(std_circ)
        std_eq = (np.product(std_circ))**(1./len(kappas))
    else:
        raise ValueError("kappa is a strange dimension")
    #print std_eq.shape, kappas.shape
    #print std_eq, std_bar
    if deg:
        return std_bar*180./np.pi, std_eq*180./np.pi
    else:
        return std_bar, std_eq

def compare_several_clusters(clusters, pub_fig = 0, alpha = 0.05,decimation=10, labels=None, filename='hello.pdf', kappa_ref_cutoff=0, plot_indices = [0,1], colours = None, markers = None, ylabel_loc = 0):
    '''
    Clusters contains a list of clusters
    Print out a comparison between two sets of clusters
    to compare datamining methods...

    SH : 7May2013
    '''
    if pub_fig:
        cm_to_inch=0.393701
        import matplotlib as mpl
        old_rc_Params = mpl.rcParams
        mpl.rcParams['font.size']=8.0
        mpl.rcParams['axes.titlesize']=8.0#'medium'
        mpl.rcParams['xtick.labelsize']=8.0
        mpl.rcParams['ytick.labelsize']=8.0
        mpl.rcParams['lines.markersize']=1.0
        mpl.rcParams['savefig.dpi']=300

    reference_cluster = clusters[0]
    clusters1 = list(set(reference_cluster.cluster_assignments))
    averages = np.average(reference_cluster.cluster_details["EM_VMM_kappas"],axis=1)
    print clusters1
    print averages
    clusters1 = np.array(clusters1)[averages > kappa_ref_cutoff]
    print clusters1
    ordering_list = [clusters1]
    string_list = []
    for test_cluster,cur_label in zip(clusters[1:],labels[1:]):
        print test_cluster.settings['method']
        #clusters1 = list(set(reference_cluster.cluster_assignments))
        clusters2 = list(set(test_cluster.cluster_assignments))
        similarity = np.zeros((len(clusters1),len(clusters2)),dtype=int)
        for i,c1 in enumerate(clusters1):
            for j,c2 in enumerate(clusters2):
                #tmp1 = (cluster1.cluster_assignments==c1)
                #similarity[i,j]=np.sum(cluster2.cluster_assignments[tmp1]==c2)
                similarity[i,j]=np.sum((reference_cluster.cluster_assignments==c1) * (test_cluster.cluster_assignments==c2))
                # np.sum((cluster1.cluster_assignments==c1) == (cluster2.cluster_assignments==c2))
        print '    %7s'%('clust') + ''.join(['%7d'%j2 for j1,j2 in enumerate(clusters2)])
        for i,clust_num in enumerate(clusters1):
            print 'ref %7d'%(clust_num,) + ''.join(['%7d'%j for j in similarity[i,:]])
            #print '%5d'.join(map(int, similarity[i,:]))
        tmp1 = float(np.sum(np.max(similarity,axis=1)))
        data_points = len(reference_cluster.cluster_assignments)
        print('correct:{}, false:{}'.format(int(tmp1), data_points-int(tmp1)))
        string_list.append([cur_label,'correct_percentage:{:.2f}'.format(tmp1/np.sum(similarity)*100)])
        print('correct_percentage:{:.2f}'.format(tmp1/np.sum(similarity)*100))
        print('false_percentage:{:.2f}'.format(100-tmp1/np.sum(similarity)*100))
        #best_match_for_c1 = np.argmax(similarity,axis=1)
        #best_match_for_c2 = np.argmax(similarity,axis=0)

        #truth = (similarity==similarity)
        order = np.zeros(similarity.shape[0],dtype=int)
        for i in range(similarity.shape[0]):
            index = np.argmax(similarity)
            row_index, col_index = np.unravel_index(similarity.argmax(), similarity.shape)
            print col_index, row_index
            #col_index = index %similarity.shape[0]
            #row_index = index/similarity.shape[0]
            order[row_index] = col_index
            similarity[row_index,:]=0
            similarity[:,col_index]=0
            print order
        ordering_list.append(order)
    n_plots = len(clusters)
    ncols = 2
    nrows = n_plots/2
    if (n_plots - (nrows * ncols))>0.01: nrows+=1
    print n_plots, ncols, nrows
    print clusters
    fig, ax = pt.subplots(ncols=ncols,nrows=nrows, sharex = True, sharey = True)
    ax = ax.flatten()
    if pub_fig:
        fig.set_figwidth(8.48*cm_to_inch)
        fig.set_figheight(8.48*(0.5*nrows)*cm_to_inch)

    instance_array = reference_cluster.feature_obj.instance_array % (2.*np.pi)
    instance_array[instance_array>np.pi]-=(2.*np.pi)
    print('####################')
    for i in string_list: print i
    print('####################')
    for test_cluster, cur_ax, order, index in zip(clusters,ax, ordering_list, range(len(ordering_list))):
        print 'hello', order
        cluster_list = list(set(test_cluster.cluster_assignments))
        n_dimensions = instance_array.shape[1]
        if (colours == None) or (markers == None):
            colours_base = ['r','k','b','y','m']
            marker_base = ['o','x','+','s','*']
            markers = []; colours = []
            for i in marker_base:
                markers.extend([i for j in colours_base])
                colours.extend(colours_base)
        for ref,cur in enumerate(order):
            #print i, colours[ref],markers[ref]
            cluster = cluster_list[order[ref]]
            current_items = test_cluster.cluster_assignments==cluster
            datapoints = instance_array[current_items,:]
            cur_ax.scatter(datapoints[::decimation,plot_indices[0]], datapoints[::decimation,plot_indices[1]],c=colours[ref],marker=markers[ref], alpha=alpha,rasterized=True,edgecolors=colours[ref])
            if labels==None:
                cur_label = test_cluster.settings['method']
            else:
                cur_label = labels[index]
        cur_ax.text(0,ylabel_loc,cur_label, horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', alpha=0.5))
            #cur_ax.plot(cluster_means[i,0],cluster_means[i,1],colours[i]+markers[i],markersize=8)
    ax[-1].set_xlim([-np.pi,np.pi])
    ax[-1].set_ylim([-np.pi,np.pi])

    fig.text(0.5, 0.01, r'$\Delta \psi_{}$'.format(plot_indices[0]), ha='center', va='center', fontsize = 10)
    fig.text(0.01, 0.5, r'$\Delta \psi_{}$'.format(plot_indices[1]), ha='center', va='center', rotation='vertical', fontsize=10)
    
    fig.subplots_adjust(hspace=0.015, wspace=0.015,left=0.10, bottom=0.10,top=0.95, right=0.95)
    fig.tight_layout()
    #fig.subplots_adjust(hspace=0.02, wspace=0.01)#,left=0.10, bottom=0.10,top=0.95, right=0.95)
    #fig.savefig('hello.eps', bbox_inches='tight', pad_inches=0)
    fig.savefig(filename, bbox_inches='tight', pad_inches=0.01)

    fig.canvas.draw(); fig.show()
    return fig, ax


def compare_two_cluster_results(cluster1, cluster2):
    '''
    Print out a comparison between two sets of clusters
    to compare datamining methods...

    SH : 7May2013
    '''
    clusters1 = list(set(cluster1.cluster_assignments))
    clusters2 = list(set(cluster2.cluster_assignments))
    similarity = np.zeros((len(clusters1),len(clusters2)),dtype=int)
    for i,c1 in enumerate(clusters1):
        for j,c2 in enumerate(clusters2):
            #tmp1 = (cluster1.cluster_assignments==c1)
            #similarity[i,j]=np.sum(cluster2.cluster_assignments[tmp1]==c2)
            similarity[i,j]=np.sum((cluster1.cluster_assignments==c1) * (cluster2.cluster_assignments==c2))
            # np.sum((cluster1.cluster_assignments==c1) == (cluster2.cluster_assignments==c2))
    print '%7s'%('clust') + ''.join(['%7d'%j2 for j1,j2 in enumerate(clusters2)])
    for i,clust_num in enumerate(clusters1):
        print '%7d'%(clust_num,) + ''.join(['%7d'%j for j in similarity[i,:]])
        #print '%5d'.join(map(int, similarity[i,:]))
    best_match_for_c1 = np.argmax(similarity,axis=1)
    best_match_for_c2 = np.argmax(similarity,axis=0)

    #print np.argmax(similarity,axis=1)

    n_clusters = len(clusters1)
    n_cols = int(math.ceil(n_clusters**0.5))
    kh_plot_item = 'kh'
    freq_plot_item = 'freq'
    if n_clusters/float(n_cols)>n_clusters/n_cols:
        n_rows = n_clusters/n_cols + 1
    else:
        n_rows = n_clusters/n_cols
    #n_rows = 4; n_cols = 4
    fig, ax = pt.subplots(nrows = n_rows, ncols = n_cols, sharex = True, sharey = True); ax = ax.flatten()
    fig2, ax2 = pt.subplots(nrows = n_rows, ncols = n_cols, sharex = True, sharey = True); ax2 = ax2.flatten()
    for i,cluster,best_match in zip(range(len(clusters1)),clusters1,best_match_for_c1):
        current_items1 = cluster1.cluster_assignments==cluster
        current_items2 = cluster2.cluster_assignments==best_match
        both = current_items2*current_items1
        if np.sum(current_items1)>10:
            ax[i].scatter((cluster1.feature_obj.misc_data_dict[kh_plot_item][current_items1]), (cluster1.feature_obj.misc_data_dict[freq_plot_item][current_items1])/1000., s=80, c='b',  marker='o', norm=None, alpha=0.02)
            ax[i].scatter((cluster2.feature_obj.misc_data_dict[kh_plot_item][current_items2]), (cluster2.feature_obj.misc_data_dict[freq_plot_item][current_items2])/1000., s=80, c='r',  marker='o', norm=None, alpha=0.02)
            ax[i].scatter((cluster1.feature_obj.misc_data_dict[kh_plot_item][both]), (cluster1.feature_obj.misc_data_dict[freq_plot_item][both])/1000., s=80, c='k',  marker='o', norm=None, alpha=0.02)
    for i,cluster,best_match in zip(range(len(clusters2)),clusters2,best_match_for_c2):
        current_items2 = cluster2.cluster_assignments==cluster
        current_items1 = cluster1.cluster_assignments==best_match
        both = current_items2*current_items1
        if np.sum(current_items1)>10:
            ax2[i].scatter((cluster2.feature_obj.misc_data_dict[kh_plot_item][current_items2]), (cluster2.feature_obj.misc_data_dict[freq_plot_item][current_items2])/1000., s=80, c='b',  marker='o', norm=None, alpha=0.02)
            ax2[i].scatter((cluster1.feature_obj.misc_data_dict[kh_plot_item][current_items1]), (cluster1.feature_obj.misc_data_dict[freq_plot_item][current_items1])/1000., s=80, c='r',  marker='o', norm=None, alpha=0.02)
            ax2[i].scatter((cluster1.feature_obj.misc_data_dict[kh_plot_item][both]), (cluster1.feature_obj.misc_data_dict[freq_plot_item][both])/1000., s=80, c='k',  marker='o', norm=None, alpha=0.02)

    ax[-1].set_xlim([0.201,0.99])
    ax[-1].set_ylim([0.1,99.9])
    fig.suptitle('blue cluster1, red best match from cluster2, black common to both')
    fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
    fig.canvas.draw(); fig.show()
    ax2[-1].set_xlim([0.201,0.99])
    ax2[-1].set_ylim([0.1,99.9])
    fig2.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
    fig2.suptitle('blue cluster2, red best match from cluster1, black common to both')
    fig2.canvas.draw(); fig2.show()
    return similarity


class feature_object():
    '''
    This is suposed to be the feature object
    SH : 6May2013
    '''
    def __init__(self, instance_array=None, misc_data_dict=None, filename = None, instance_array_amps = None):#, misc_data_labels):
        '''
        feature_object... this contains all of the raw data that is a 
        result of feature extraction. It can be initialised by passing
        an instance_array and misc_data_dict dictionary, or alternatively,
        the filename of a pickle file that was saved with 
        feature_object.dump_data()
        '''
        if instance_array is None and filename is not None:
            self.load_data(filename)
        else:
            self.instance_array = instance_array
            if instance_array_amps is not None: self.instance_array_amps = instance_array_amps
            self.misc_data_dict = misc_data_dict
            self.clustered_objects = []

    def cluster(self,**kwargs):
        '''This method will perform clustering using one of the
        following methods: k-means (scikit-learn), Expectation
        maximising using a Gaussian mixture model EM_GMM
        (scikit-learn) k_means_periodic (SH implementation) Expecation
        maximising using a von Mises mixture model EM_VMM (SH
        implementation)

        **kwargs: method : To determine which clustering algorithm to
        use can be : k-means, EMM_GMM, k_means_periodic, EM_VMM

        Other kwargs to overide the following default settings for
        each clustering algorithmn

        'k_means': {'n_clusters':9, 'sin_cos':1,
        'number_of_starts':30}, 'EM_GMM' : {'n_clusters':9,
        'sin_cos':1, 'number_of_starts':30}, 'k_means_periodic' :
        {'n_clusters':9, 'number_of_starts':10, 'n_cpus':1,
        'distance_calc':'euclidean','convergence_diff_cutoff': 0.2,
        'iterations': 40, 'decimal_roundoff':2}, 'EM_VMM' :
        {'n_clusters':9, 'n_iterations':20, 'n_cpus':1}}

        returns a cluster object that also gets appended to the
        self.clustered_objects list

        SH: 6May2013
        '''
        self.clustered_objects.append(clusterer_wrapper(self,**kwargs))
        return self.clustered_objects[-1]


    def dump_data(self,filename):
        '''
        This is saves the important parts of the clustering data
        It does not save the object itself!!!

        The idea here is that we can save the data, and when we reload it,
        we have access to any new features.

        SH: 8May2013
        '''
        dump_dict = {}
        dump_dict['instance_array'] = self.instance_array
        dump_dict['misc_data_dict'] = self.misc_data_dict
        dump_dict['clustered_objects']={}
        clust_objs = dump_dict['clustered_objects']
        for i,tmp_clust in enumerate(self.clustered_objects):
            clust_objs[i]={}
            clust_objs[i]['settings']=tmp_clust.settings
            clust_objs[i]['cluster_assignments']=tmp_clust.cluster_assignments
            clust_objs[i]['cluster_details']=tmp_clust.cluster_details
        pickle.dump(dump_dict,file(filename,'w'))

    def load_data(self,filename):
        '''
        This is for loading saved clustering data
        SH: 8May2013
        '''
        dump_dict = pickle.load(file(filename,'r'))
        self.instance_array = dump_dict['instance_array']
        self.misc_data_dict = dump_dict['misc_data_dict']
        self.clustered_objects = []
        clust_objs = dump_dict['clustered_objects']
        for i in clust_objs.keys():
            tmp = clustering_object()
            tmp.settings = clust_objs[i]['settings']
            tmp.cluster_assignments = clust_objs[i]['cluster_assignments']
            tmp.cluster_details = clust_objs[i]['cluster_details']
            tmp.feature_obj = self
            self.clustered_objects.append(tmp)

    def print_cluster_details(self,):
        for i,clust in enumerate(self.clustered_objects):
            print i, clust.settings
            

class clustering_object():
    '''Generic clustering_object, this will have the following
    attributes instance_array : array of phase differences

    SH : 6May2013 '''

    def mode_num_analysis(self, array = 'HMA', other_array = False, other_array_name = None, boozer_files_location='/home/srh112/code/python/h1_eq_generation/results_jason6/'):
        '''Supposed to fit a whole bunch of modes to the data for each cluster
        array can be one of HMA, PMA1, PMA2 or PMA1_reduced

        if other_array is True then the data is obtained from 
          self.feature_obj.misc_data_dict[other_array_name]
        instead of the main instance array. Make sure this data exists and is
        included using 'other_arrays' when extracting the data

        Certain probes are excluded because they are dead or have poor signal 
        Requires the magnetics part of the h1 module for the probe details
        SRH: 7Sept2014
        '''
        
        import h1.diagnostics.magnetics as mag
        if array == 'HMA':
            self.arr = mag.HMA()
            mask = np.ones(self.arr.cart_x.shape[0], dtype = bool)
            #exclude the naked coil
            mask[0] = False
        elif array == 'PMA1':
            self.arr = mag.PMA1()
            mask = np.ones(self.arr.cart_x.shape[0], dtype = bool)
            for i in [5,6,11,12,13,14,19,20]:
                mask[i-1] = False
        elif array == 'PMA2':
            self.arr = mag.PMA2()
            mask = np.ones(self.arr.cart_x.shape[0], dtype = bool)
        elif array == 'PMA1_reduced':
            print 'hello'
            self.arr = mag.PMA1()
            mask = np.ones(self.arr.cart_x.shape[0], dtype = bool)
            for i in [5,6,11,12,13,14,19,20]:
                mask[i-1] = False
            #self.arr = mag.PMA1_reduced()
            #mask = np.ones(self.arr.cart_x.shape[0], dtype = bool)
        cluster_list = list(set(self.cluster_assignments))
        n_clusters = len(cluster_list)
        misc_data_dict = self.feature_obj.misc_data_dict
        if not hasattr(self,'coil_locs'):
            self.coil_locs = {'HMA':{},'PMA1':{},'PMA2':{}, 'PMA1_reduced':{}}
        #cluster_mu = self.cluster_details['EM_VMM_means']
        fig, ax = make_grid_subplots(n_clusters, sharex = True, sharey = True)
        fig2, ax2 = make_grid_subplots(n_clusters, sharex = True, sharey = True)
        fig3, ax3 = make_grid_subplots(n_clusters, sharex = True, sharey = True)
        foo = self.feature_obj.misc_data_dict[other_array_name] if other_array else self.feature_obj.misc_data_dict['mirnov_data']
        #foo = self.feature_obj.misc_data_dict[other_array_name]
        foo = foo/np.abs(foo)
        foo2 = foo / np.sum(foo, axis = 1)[:,np.newaxis]
        cur_data = foo * np.sum(foo2,axis = 1)[:,np.newaxis]
        #cur_data = np.zeros((foo.shape[0], foo.shape[1]-1), dtype = complex)
        #for i in range(1,foo.shape[1]): cur_data[:,i-1] = foo[:,i]/foo[:,i-1]
        self.cluster_mode_fits = {}
        for i, cluster in enumerate(cluster_list):
            current_items = self.cluster_assignments==cluster
            #tmp_data = np.exp(1j*(np.cumsum(loc_tmp)))
            mean, kappa, foo = EM_VMM_calc_best_fit(cur_data[current_items,:], N=np.sum(current_items))
            loc_tmp = np.angle(np.sum(cur_data[current_items,:], axis = 0))
            # else:
            #     #tmp_data = np.exp(1j*(np.cumsum(loc_tmp)))
            #     mean, kappa, foo = EM_VMM_calc_best_fit(cur_data[current_items,:], N=np.sum(current_items))
            #     loc_tmp = np.angle(np.sum(cur_data[current_items,:], axis = 0))
            #     #loc_tmp = mean
            #     print 'MEAN: ', mean, kappa, foo
            #     #phases
            #     #cur_data = np.exp(1j*self.feature_obj.instance_array[current_items,:])
            #     #loc_tmp = cluster_mu[cluster]#[:14]
            if np.sum(current_items)>10:
                kh_ave_round = 0.5 * np.round(2.0 * (np.mean(misc_data_dict['kh'][current_items])*10)) / 10.
                kh_ave_round = np.max([np.min([kh_ave_round, 0.9]),0.1])
                print kh_ave_round
                if '{:.3f}'.format(kh_ave_round) in self.coil_locs[array].keys():
                    self.arr.loc_boozer_coords(filename = None, **self.coil_locs[array]['{:.3f}'.format(kh_ave_round)])
                    print 'using preused values'
                else:
                    avail = np.array([0.33,0.37,0.44,0.63,0.69,0.83])
                    loc = np.argmin(np.abs(kh_ave_round - avail))
                    #filename  = '/home/srh112/code/python/h1_eq_generation/results7/kh%.3f-kv1.000fixed/boozmn_wout_kh%.3f-kv1.000fixed.nc'%(kh_ave_round, kh_ave_round)
                    filename  = '{}/kh{:.3f}-kv1.000fixed/boozmn_wout_kh{:.3f}-kv1.000fixed.nc'.format(boozer_files_location, avail[loc], avail[loc])
                    print filename, kh_ave_round
                    self.arr.loc_boozer_coords(filename = filename)
                    self.coil_locs[array]['{:.3f}'.format(kh_ave_round)] = {}
                    for tmp_ind in ['boozer_phi', 'boozer_theta','distance']:self.coil_locs[array]['{:.3f}'.format(kh_ave_round)][tmp_ind] = copy.deepcopy(getattr(self.arr, tmp_ind))
                
                tmp_data = np.exp(1j*(np.cumsum(loc_tmp)))
                data = np.mean(np.real(cur_data[current_items,:]), axis = 0) + 1j*np.mean(np.imag(cur_data[current_items,:]), axis = 0)
                data = data/np.abs(data)
                #data = np.zeros(tmp_data.shape[0]+1,dtype = complex)
                #data[1:] = +tmp_data
                #data = data[::-1]
                self.arr.perform_fit(data, mask = mask, inc_phi = True)
                #ax2[cluster].plot(loc_tmp)
                self.cluster_mode_fits[cluster] = copy.deepcopy(self.arr.vals)
                x_ax = np.unwrap(np.deg2rad(self.arr.boozer_phi)) if array == 'HMA' else np.unwrap(np.deg2rad(self.arr.boozer_theta))
                self.arr.plot_fig(ax = ax[cluster], ax2 = ax2[cluster], mask = mask, ax2_xaxis = x_ax[mask], ax3 = ax3[cluster])
                ax2[cluster].plot(x_ax[mask], np.real(data), '-bo')
                ax2[cluster].plot(x_ax[mask], np.imag(data), '-ro')
                ax3[cluster].plot(np.real(data), np.imag(data), 'r.')
                for j in range(data.shape[0]): ax3[cluster].text(np.real(data[j]), np.imag(data[j]), str(j+1))
                ax2[cluster].grid()

        self.cluster_mode_fits['m'] = copy.deepcopy(self.arr.m_rec)
        self.cluster_mode_fits['n'] = copy.deepcopy(self.arr.n_rec)
        if not hasattr(self,'all_cluster_mode_fits'):
            print('all_cluster_mode_fits does not exist, creating it')
            self.all_cluster_mode_fits = {}
        self.all_cluster_mode_fits[array] = copy.deepcopy(self.cluster_mode_fits)

        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        ax[0].set_xlim([np.min(self.arr.m_rec), np.max(self.arr.m_rec)])
        ax[0].set_ylim([np.min(self.arr.n_rec), np.max(self.arr.n_rec)])
        ax2[0].set_xlim([np.min(x_ax),np.max(x_ax)])
        ax2[0].set_ylim([-1.1,1.1])
        fig.canvas.draw(); fig.show()
        fig2.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig2.canvas.draw(); fig2.show()
        ax3[0].set_xlim([np.min(self.arr.m_rec), np.max(self.arr.m_rec)])
        ax3[0].set_xlim([-1, 1])
        ax3[0].set_ylim([-1, 1])
        fig3.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig3.canvas.draw(); fig2.show()



    def plot_kh_freq_all_clusters(self,color_by_cumul_phase = 1):
        '''plot kh vs frequency for each cluster - i.e looking for
        whale tails The colouring of the points is based on the total
        phase along the array i.e a 1D indication of the clusters

        SH: 9May2013

        '''
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'').replace('_','-')
        kh_plot_item = 'kh'
        freq_plot_item = 'freq'
        cluster_list = list(set(self.cluster_assignments))
        n_clusters = len(cluster_list)
        fig_kh, ax_kh = make_grid_subplots(n_clusters, sharex = True, sharey = True)
        misc_data_dict = self.feature_obj.misc_data_dict
        if color_by_cumul_phase:
            instance_array2 = modtwopi(self.feature_obj.instance_array, offset=0)
            max_lim = -1*2.*np.pi; min_lim = (-3.*2.*np.pi)
            total_phase = np.sum(instance_array2,axis=1)
            print np.max(total_phase), np.min(total_phase)
            total_phase = np.clip(total_phase,min_lim, max_lim)
        for cluster in cluster_list:
            current_items = self.cluster_assignments==cluster
            if np.sum(current_items)>10:
                if color_by_cumul_phase:
                    ax_kh[cluster].scatter((misc_data_dict[kh_plot_item][current_items]), (misc_data_dict[freq_plot_item][current_items])/1000., s=80, c=total_phase[current_items], vmin = min_lim, vmax = max_lim, marker='o', cmap='jet', norm=None, alpha=0.2)
                else:
                    ax_kh[cluster].scatter((misc_data_dict[kh_plot_item][current_items]), (misc_data_dict[freq_plot_item][current_items])/1000, s=100, c='b', marker='o', cmap=None, norm=None)
                ax_kh[cluster].legend(loc='best')
        ax_kh[-1].set_xlim([0.201,0.99])
        ax_kh[-1].set_ylim([0.1,199.9])
        fig_kh.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig_kh.suptitle(suptitle,fontsize=8)
        fig_kh.canvas.draw(); fig_kh.show()
        return fig_kh, ax_kh


    def plot_time_freq_all_clusters(self,color_by_cumul_phase = 1):
        '''plot kh vs frequency for each cluster - i.e looking for whale tails
        The colouring of the points is based on the total phase along the array
        i.e a 1D indication of the clusters

        SH: 9May2013

        '''
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        time_plot_item = 'time'
        freq_plot_item = 'freq'
        cluster_list = list(set(self.cluster_assignments))
        n_clusters = len(cluster_list)
        fig_kh, ax_kh = make_grid_subplots(n_clusters, sharex = True, sharey = True)
        misc_data_dict = self.feature_obj.misc_data_dict
        if color_by_cumul_phase:
            instance_array2 = modtwopi(self.feature_obj.instance_array, offset=0)
            max_lim = -1*2.*np.pi; min_lim = (-3.*2.*np.pi)
            total_phase = np.sum(instance_array2,axis=1)
            print np.max(total_phase), np.min(total_phase)
            total_phase = np.clip(total_phase,min_lim, max_lim)
        for cluster in cluster_list:
            current_items = self.cluster_assignments==cluster
            if np.sum(current_items)>10:
                if color_by_cumul_phase:
                    ax_kh[cluster].scatter((misc_data_dict[time_plot_item][current_items]), (misc_data_dict[freq_plot_item][current_items]), s=20, c=total_phase[current_items], vmin = min_lim, vmax = max_lim, marker='o', cmap='jet', norm=None, alpha=0.8)
                else:
                    ax_kh[cluster].scatter((misc_data_dict[time_plot_item][current_items]), (misc_data_dict[freq_plot_item][current_items]), s=100, c='b', marker='o', cmap=None, norm=None)
                ax_kh[cluster].legend(loc='best')
        #ax_kh[-1].set_xlim([0.,0.1])
        #ax_kh[-1].set_ylim([0.,150.])
        fig_kh.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig_kh.suptitle(suptitle,fontsize=8)
        fig_kh.canvas.draw(); fig_kh.show()
        return fig_kh, ax_kh

    def fit_vonMises(self,):
        instance_array = self.feature_obj.instance_array
        mu_list = np.ones((len(set(self.cluster_assignments)),self.feature_obj.instance_array.shape[1]),dtype=float)
        kappa_list = mu_list*1.
        self.cluster_details['EM_VMM_means'], self.cluster_details['EM_VMM_kappas'],tmp1,tmp2 = _EM_VMM_maximisation_step_hard(mu_list, kappa_list, self.feature_obj.instance_array, self.cluster_assignments)

    def plot_VM_distributions(self,):
        '''Plot the vonMises distributions for each dimension for each cluster
        Also plot the histograms - these are overlayed with dashed lines

        SH: 9May2013
        '''
        #Plot the two distributions over each other to check for goodness of fit
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        cluster_list = list(set(self.cluster_assignments))
        n_clusters = np.max([len(cluster_list),np.max(cluster_list)])
        fig, ax = make_grid_subplots(n_clusters, sharex = True, sharey = True)
        delta = 300
        x = np.linspace(-np.pi, np.pi, delta)
        instance_array = self.feature_obj.instance_array
        try:
            cluster_mu = self.cluster_details['EM_VMM_means']
            cluster_kappa = self.cluster_details['EM_VMM_kappas']
        except KeyError:
            print 'EM_VMM cluster details not available - calculating them'
            self.fit_vonMises()
        cluster_mu = self.cluster_details['EM_VMM_means']
        cluster_kappa = self.cluster_details['EM_VMM_kappas']
        for cluster in cluster_list:
            current_items = self.cluster_assignments==cluster
            if np.sum(current_items)>10:
                for dimension in range(instance_array.shape[1]):
                    #print cluster, dimension
                    kappa_tmp = cluster_kappa[cluster][dimension]
                    loc_tmp = cluster_mu[cluster][dimension]
                    fit_dist_EM = vonmises(kappa_tmp,loc_tmp)
                    Z_EM = fit_dist_EM.pdf(x)
                    if np.sum(np.isnan(Z_EM))==0:
                        tmp = ax[cluster].plot(x,Z_EM,'-')
                        current_color = tmp[0].get_color()
                        ax[cluster].text(x[np.argmax(Z_EM)],np.max(Z_EM),str(dimension))
                    bins = np.linspace(-np.pi,np.pi,360)
                    histogram_data = (instance_array[current_items,dimension]) %(2.*np.pi)
                    histogram_data[histogram_data>np.pi]-=(2.*np.pi)
                    tmp3 = np.histogram(histogram_data,bins = bins,range=[-np.pi,np.pi])
                    dx = tmp3[1][1]-tmp3[1][0]
                    integral = np.sum(dx*tmp3[0])
                    if np.sum(np.isnan(tmp3[0]/integral))==0:
                        ax[cluster].plot(tmp3[1][:-1]+dx/2, tmp3[0]/integral, marker=',',linestyle="--",color=current_color)
        ax[-1].set_xlim([-np.pi,np.pi])
        fig.subplots_adjust(hspace=0, wspace=0)
        fig.suptitle(suptitle,fontsize=8)
        fig.canvas.draw();fig.show()
        return fig, ax


    def plot_GMM_GMM_distributions(self,):
        '''Plot the vonMises distributions for each dimension for each cluster
        Also plot the histograms - these are overlayed with dashed lines

        SH: 9May2013
        '''
        #Plot the two distributions over each other to check for goodness of fit
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        cluster_list = list(set(self.cluster_assignments))
        n_clusters = np.max([len(cluster_list),np.max(cluster_list)])
        fig_re, ax_re = make_grid_subplots(n_clusters, sharex = True, sharey = True)
        fig_im, ax_im = make_grid_subplots(n_clusters, sharex = True, sharey = True)
        delta = 300
        x = np.linspace(-np.pi, np.pi, delta)
        instance_array = self.feature_obj.instance_array
        cluster_mu_re = self.cluster_details['EM_GMM_means_re']
        cluster_sigma_re = self.cluster_details['EM_GMM_variances_re']
        cluster_mu_im = self.cluster_details['EM_GMM_means_im']
        cluster_sigma_im = self.cluster_details['EM_GMM_variances_im']

        for cluster in cluster_list:
            current_items = self.cluster_assignments==cluster
            print 'hello'
            if np.sum(current_items)>10:
                for dimension in range(cluster_mu_re.shape[1]):
                    Z_EM_re = norm(loc=cluster_mu_re[cluster][dimension], scale=np.sqrt(cluster_sigma_re[cluster][dimension])).pdf(x)
                    Z_EM_im = norm(loc=cluster_mu_im[cluster][dimension], scale=np.sqrt(cluster_sigma_im[cluster][dimension])).pdf(x)
                    if np.sum(np.isnan(Z_EM_re))==0:
                        tmp = ax_re[cluster].plot(x,Z_EM_re,'-')
                        current_color = tmp[0].get_color()
                        ax_re[cluster].text(x[np.argmax(Z_EM_re)],np.max(Z_EM_re),str(dimension))
                    if np.sum(np.isnan(Z_EM_im))==0:
                        tmp = ax_im[cluster].plot(x,Z_EM_im,'-')
                        ax_im[cluster].text(x[np.argmax(Z_EM_im)],np.max(Z_EM_im),str(dimension))
                        current_color = tmp[0].get_color()
                    #bins = np.linspace(-np.pi,np.pi,360)
                    #histogram_data = (instance_array[current_items,dimension]) %(2.*np.pi)
                    #histogram_data[histogram_data>np.pi]-=(2.*np.pi)
                    #tmp3 = np.histogram(histogram_data,bins = bins,range=[-np.pi,np.pi])
                    #dx = tmp3[1][1]-tmp3[1][0]
                    #integral = np.sum(dx*tmp3[0])
                    #if np.sum(np.isnan(tmp3[0]/integral))==0:
                    #    ax[cluster].plot(tmp3[1][:-1]+dx/2, tmp3[0]/integral, marker=',',linestyle="--",color=current_color)
        ax_re[-1].set_xlim([-np.pi,np.pi])
        ax_re[-1].set_ylim([0,4])
        ax_im[-1].set_xlim([-np.pi,np.pi])
        ax_im[-1].set_ylim([0,4])
        fig_im.subplots_adjust(hspace=0, wspace=0)
        #fig_im.suptitle(suptitle,fontsize=8)
        fig_im.canvas.draw();fig_im.show()
        fig_re.subplots_adjust(hspace=0, wspace=0)
        #fig_re.suptitle(suptitle,fontsize=8)
        fig_re.canvas.draw();fig_im.show()
        #return fig, ax


    def plot_dimension_histograms(self,pub_fig = 0, filename='plot_dim_hist.pdf',specific_dimensions = None, extra_txt_labels = '', label_loc = [-2,1.5], ylim = None):
        '''For each dimension in the data set, plot the histogram of the phase differences
        Overlay the vonMises mixture model along with the individual vonMises distributions 
        from each cluster

        SH: 9May2013
        '''
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        instance_array = self.feature_obj.instance_array
        dimensions = instance_array.shape[1]
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')

        if pub_fig:
            cm_to_inch=0.393701
            import matplotlib as mpl
            old_rc_Params = mpl.rcParams
            mpl.rcParams['font.size']=8.0
            mpl.rcParams['axes.titlesize']=8.0#'medium'
            mpl.rcParams['xtick.labelsize']=8.0
            mpl.rcParams['ytick.labelsize']=8.0
            mpl.rcParams['lines.markersize']=1.0
            mpl.rcParams['savefig.dpi']=300
        if specific_dimensions == None:
            specific_dimensions = range(instance_array.shape[1])
        fig, ax = make_grid_subplots(len(specific_dimensions), sharex = True, sharey = True)
        if pub_fig:
            fig.set_figwidth(8.48*cm_to_inch)
            fig.set_figheight(8.48*0.8*cm_to_inch)
        for i,dim in enumerate(specific_dimensions):
            histogram_data = (instance_array[:,dim]) %(2.*np.pi)
            histogram_data[histogram_data>np.pi]-=(2.*np.pi)
            ax[i].hist(histogram_data,bins=180,normed=True,histtype='stepfilled',range=[-np.pi,np.pi])
        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        #fig.canvas.draw(); fig.show()
        if self.cluster_assignments!=None:
            cluster_list = list(set(self.cluster_assignments))
        delta = 300
        x = np.linspace(-np.pi, np.pi, delta)
        cluster_prob_list = []
        for cluster in cluster_list:
            cluster_prob_list.append(float(np.sum(self.cluster_assignments==cluster))/float(len(self.cluster_assignments)))
        #print cluster_prob_list
        try:
            cluster_mu = self.cluster_details['EM_VMM_means']
            cluster_kappa = self.cluster_details['EM_VMM_kappas']
        except KeyError:
            print 'EM_VMM cluster details not available - calculating them'
            self.fit_vonMises()
        cluster_mu = self.cluster_details['EM_VMM_means']
        cluster_kappa = self.cluster_details['EM_VMM_kappas']
        text_labels = []
        for i,dimension in enumerate(specific_dimensions):
            cluster_sum = x*0
            for cluster, cluster_prob in zip(cluster_list, cluster_prob_list):
                current_items = (self.cluster_assignments==cluster)
                kappa_tmp = cluster_kappa[cluster][dimension]
                loc_tmp = cluster_mu[cluster][dimension]
                fit_dist_EM = vonmises(kappa_tmp,loc_tmp)
                Z_EM = cluster_prob * fit_dist_EM.pdf(x)
                cluster_sum = Z_EM + cluster_sum
                if pub_fig:
                    tmp = ax[i].plot(x,Z_EM,'-',linewidth=0.8)
                else:
                    tmp = ax[i].plot(x,Z_EM,'-',linewidth=2)
            if pub_fig:
                tmp = ax[i].plot(x,cluster_sum,'-',linewidth=2)
            else:
                tmp = ax[i].plot(x,cluster_sum,'-',linewidth=4)
            print '{area},'.format(area = np.sum(cluster_sum*(x[1]-x[0]))),
            ax[i].text(label_loc[0], label_loc[1],r'$\Delta \psi_%d$ '%(dimension+1,) + extra_txt_labels, fontsize = 8)#,bbox=dict(facecolor='white', alpha=0.5))
            ax[i].locator_params(nbins=7)
        print ''
        if pub_fig:
            ax[0].set_xlim([-np.pi, np.pi])
            if ylim!=None:
                ax[0].set_ylim(ylim)
            fig.text(0.5, 0.01, r'$\Delta \psi$ (rad)', ha='center', va='center', fontsize = 10)
            fig.text(0.01, 0.5, 'Probability density', ha='center', va='center', rotation='vertical', fontsize=10)
            fig.subplots_adjust(hspace=0.015, wspace=0.015,left=0.10, bottom=0.10,top=0.95, right=0.95)
            fig.tight_layout()
            fig.savefig(filename, bbox_inches='tight', pad_inches=0.01)
        ax[-1].set_xlim([-np.pi,np.pi])
        fig.suptitle(suptitle.replace('_','\char`_'),fontsize = 8)
        fig.canvas.draw(); fig.show()
        return fig, ax



    def plot_dimension_histograms_GMM_GMM(self,pub_fig = 0, filename='plot_dim_hist.pdf',specific_dimensions = None, extra_txt_labels = '', label_loc = [-2,1.5], ylim = None):
        '''For each dimension in the data set, plot the histogram of the real and imag part of the measurements
        overlay the GMM's - used for the GMM-GMM clustering method

        SRH: 18May2014
        '''
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        cluster_mu = self.cluster_details['EM_GMM_means_re'] + 1j*self.cluster_details['EM_GMM_means_im']
        cluster_sigma = self.cluster_details['EM_GMM_variances_re'] + 1j*self.cluster_details['EM_GMM_variances_im']
        dimensions = cluster_mu.shape[1]

        instance_array_amps = self.feature_obj.misc_data_dict['mirnov_data']
        tmp = np.zeros((instance_array_amps.shape[0], instance_array_amps.shape[1]-1),dtype=complex)
        tmp = instance_array_amps/np.sum(instance_array_amps, axis = 1)[:,np.newaxis]
        #for i in range(1,instance_array_amps.shape[1]): tmp[:,i-1] = instance_array_amps[:,i]/instance_array_amps[:,i-1]
        if specific_dimensions == None: specific_dimensions = range(dimensions)
        fig_re, ax_re = make_grid_subplots(len(specific_dimensions), sharex = True, sharey = True)
        fig_im, ax_im = make_grid_subplots(len(specific_dimensions), sharex = True, sharey = True)
        for i,dim in enumerate(specific_dimensions):
            ax_re[i].hist(np.real(tmp[:,dim]), bins=180,normed=True,histtype='stepfilled',range=[-np.pi,np.pi])
            ax_im[i].hist(np.imag(tmp[:,dim]), bins=180,normed=True,histtype='stepfilled',range=[-np.pi,np.pi])

        if self.cluster_assignments!=None: cluster_list = list(set(self.cluster_assignments))
        x = np.linspace(-np.pi, np.pi, 300)
        cluster_prob_list = []
        for cluster in cluster_list:
            cluster_prob_list.append(float(np.sum(self.cluster_assignments==cluster))/float(len(self.cluster_assignments)))
        for i, dimension in enumerate(specific_dimensions):
            for ax_cur, op in zip([ax_re,ax_im],[np.real, np.imag]):
                cluster_sum = x*0
                for cluster, cluster_prob in zip(cluster_list, cluster_prob_list):
                    Z_EM = cluster_prob * norm(loc=op(cluster_mu[cluster][dimension]), scale=np.sqrt(op(cluster_sigma[cluster][dimension]))).pdf(x)
                    cluster_sum += Z_EM
                    tmp = ax_cur[i].plot(x,Z_EM,'-',linewidth=0.8)
                tmp = ax_cur[i].plot(x,cluster_sum,'-',linewidth=2)
                print '{area},'.format(area = np.sum(cluster_sum*(x[1]-x[0]))),
                ax_cur[i].text(label_loc[0], label_loc[1],r'$\Delta \psi_%d$ '%(dimension+1,) + extra_txt_labels, fontsize = 8)#,bbox=dict(facecolor='white', alpha=0.5))
                ax_cur[i].locator_params(nbins=7)
        print ''
        for ax_cur, fig_cur in zip([ax_re, ax_im],[fig_re, fig_im]):
            ax_cur[-1].set_xlim([-np.pi,np.pi])
            ax_cur[-1].set_ylim([0,1.3])
            fig_cur.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
            fig_cur.suptitle(suptitle.replace('_','\char`_'),fontsize = 8)
            fig_cur.canvas.draw(); fig_cur.show()


    def plot_dimension_histograms_VMM_GMM(self,pub_fig = 0, filename='plot_dim_hist.pdf',specific_dimensions = None, extra_txt_labels = '', label_loc = [-2,1.5], ylim = None):
        '''For each dimension in the data set, plot the histogram of the real and imag part of the measurements
        overlay the GMM's - used for the GMM-GMM clustering method

        SRH: 18May2014
        '''
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        cluster_GMM_mu = self.cluster_details['EM_GMM_means']
        cluster_GMM_sigma = self.cluster_details['EM_GMM_std']
        dimensions = cluster_GMM_mu.shape[1]

        instance_array_amps = self.feature_obj.misc_data_dict['mirnov_data']
        tmp = np.zeros((instance_array_amps.shape[0], instance_array_amps.shape[1]-1),dtype=complex)
        for i in range(1,instance_array_amps.shape[1]): tmp[:,i-1] = instance_array_amps[:,i]/instance_array_amps[:,i-1]
        if specific_dimensions == None: specific_dimensions = range(dimensions)
        fig_ang, ax_ang = make_grid_subplots(len(specific_dimensions), sharex = True, sharey = True)
        fig_abs, ax_abs = make_grid_subplots(len(specific_dimensions), sharex = True, sharey = True)
        amp_vals = np.abs(tmp)
        amp_vals[np.angle(tmp)<0]*= (-1)
        for i,dim in enumerate(specific_dimensions):
            ax_abs[i].hist(amp_vals[:,dim], bins=180,normed=True,histtype='stepfilled',range=[-np.pi,np.pi])
            ax_ang[i].hist(np.angle(tmp[:,dim]), bins=180,normed=True,histtype='stepfilled',range=[-np.pi,np.pi])

        if self.cluster_assignments!=None: cluster_list = list(set(self.cluster_assignments))
        x = np.linspace(-np.pi, np.pi, 300)
        cluster_prob_list = []
        for cluster in cluster_list:
            cluster_prob_list.append(float(np.sum(self.cluster_assignments==cluster))/float(len(self.cluster_assignments)))
        for i, dimension in enumerate(specific_dimensions):
            #for ax_cur, op in zip([ax_re,ax_im],[np.real, np.imag]):
            #for ax_cur, op in zip([ax_ang,ax_abs],[np.angle, np.abs]):
            for ax_cur, op in zip([ax_abs],[np.abs]):
                cluster_sum = x*0
                for cluster, cluster_prob in zip(cluster_list, cluster_prob_list):
                    Z_EM = cluster_prob * norm(loc=cluster_GMM_mu[cluster][dimension], scale=cluster_GMM_sigma[cluster][dimension]).pdf(x)
                    cluster_sum += Z_EM
                    tmp = ax_cur[i].plot(x,Z_EM,'-',linewidth=0.8)
                tmp = ax_cur[i].plot(x,cluster_sum,'-',linewidth=2)
                print '{area},'.format(area = np.sum(cluster_sum*(x[1]-x[0]))),
                ax_cur[i].text(label_loc[0], label_loc[1],r'$\Delta \psi_%d$ '%(dimension+1,) + extra_txt_labels, fontsize = 8)#,bbox=dict(facecolor='white', alpha=0.5))
                ax_cur[i].locator_params(nbins=7)
        print ''
        #for ax_cur, fig_cur in zip([ax_re, ax_im],[fig_re, fig_im]):
        for ax_cur, fig_cur in zip([ax_ang, ax_abs],[fig_ang, fig_abs]):
            ax_cur[-1].set_xlim([-np.pi,np.pi])
            ax_cur[-1].set_ylim([0,1.3])
            fig_cur.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
            fig_cur.suptitle(suptitle.replace('_','\char`_'),fontsize = 8)
            fig_cur.canvas.draw(); fig_cur.show()



    def plot_dimension_histograms_amps(self,pub_fig = 0, filename='plot_dim_hist.pdf',specific_dimensions = None):
        '''For each dimension in the data set, plot the histogram of the phase differences
        Overlay the vonMises mixture model along with the individual vonMises distributions 
        from each cluster

        SH: 9May2013
        '''
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        instance_array_amps = np.abs(self.feature_obj.misc_data_dict['mirnov_data'])
        norm_factor = np.sum(instance_array_amps,axis=1)
        instance_array_amps = instance_array_amps/norm_factor[:,np.newaxis]
        dimensions = instance_array_amps.shape[1]
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')

        if pub_fig:
            cm_to_inch=0.393701
            import matplotlib as mpl
            old_rc_Params = mpl.rcParams
            mpl.rcParams['font.size']=8.0
            mpl.rcParams['axes.titlesize']=8.0#'medium'
            mpl.rcParams['xtick.labelsize']=8.0
            mpl.rcParams['ytick.labelsize']=8.0
            mpl.rcParams['lines.markersize']=1.0
            mpl.rcParams['savefig.dpi']=300
        if specific_dimensions == None:
            specific_dimensions = range(instance_array_amps.shape[1])
        fig, ax = make_grid_subplots(len(specific_dimensions), sharex = True, sharey = True)
        if pub_fig:
            fig.set_figwidth(8.48*cm_to_inch)
            fig.set_figheight(8.48*0.8*cm_to_inch)
        for i,dim in enumerate(specific_dimensions):
            histogram_data = instance_array_amps[:,dim]
            ax[i].hist(histogram_data,bins=180,normed=True,histtype='stepfilled',range=[0,1])
        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig.canvas.draw(); fig.show()
        cluster_list = list(set(self.cluster_assignments))
        delta = 300
        x = np.linspace(0, 1, delta)
        cluster_prob_list = []
        for cluster in cluster_list:
            cluster_prob_list.append(float(np.sum(self.cluster_assignments==cluster))/float(len(self.cluster_assignments)))

        cluster_means = self.cluster_details['EM_GMM_means']
        cluster_std = self.cluster_details['EM_GMM_std']
        for i,dimension in enumerate(specific_dimensions):
            cluster_sum = x*0
            for cluster, cluster_prob in zip(cluster_list, cluster_prob_list):
                current_items = (self.cluster_assignments==cluster)
                std_tmp = cluster_std[cluster][dimension]
                mean_tmp = cluster_means[cluster][dimension]
                fit_dist_GM = norm(loc=mean_tmp,scale=std_tmp)
                #fit_dist_GM = norm(std_tmp,mean_tmp)
                Z_GM = cluster_prob * fit_dist_GM.pdf(x)
                cluster_sum = Z_GM + cluster_sum
                if pub_fig:
                    tmp = ax[i].plot(x,Z_GM,'-',linewidth=0.8)
                else:
                    tmp = ax[i].plot(x,Z_GM,'-',linewidth=2)
            if pub_fig:
                tmp = ax[i].plot(x,cluster_sum,'-',linewidth=2)
            else:
                tmp = ax[i].plot(x,cluster_sum,'-',linewidth=4)
            print '{area},'.format(area = np.sum(cluster_sum*(x[1]-x[0]))),
        print ''
        if pub_fig:
            ax[0].set_xlim([-np.pi, np.pi])
            fig.text(0.5, 0.01, 'Amp', ha='center', va='center', fontsize = 10)
            fig.text(0.01, 0.5, 'Probability density', ha='center', va='center', rotation='vertical', fontsize=10)
            fig.subplots_adjust(hspace=0.015, wspace=0.015,left=0.10, bottom=0.10,top=0.95, right=0.95)
            fig.tight_layo()
            fig.savefig(filename, bbox_inches='tight', pad_inches=0.01)
        ax[-1].set_xlim([0,1])
        fig.suptitle(suptitle,fontsize = 8)
        fig.canvas.draw(); fig.show()
        return fig, ax



    def plot_phase_vs_phase(self,pub_fig = 0, kappa_cutoff=None, filename = 'phase_vs_phase.pdf',compare_dimensions=None, kappa_ave_cutoff=0, plot_means = 0, alpha = 0.05, decimation = 1, xlabel_loc = 0, ylabel_loc = 0, combine_factor = 1):
        '''
        SH: 9May2013

        '''
        if pub_fig:
            cm_to_inch=0.393701
            import matplotlib as mpl
            old_rc_Params = mpl.rcParams
            mpl.rcParams['font.size']=8.0
            mpl.rcParams['axes.titlesize']=8.0#'medium'
            mpl.rcParams['xtick.labelsize']=8.0
            mpl.rcParams['ytick.labelsize']=8.0
            mpl.rcParams['lines.markersize']=1.0
            mpl.rcParams['savefig.dpi']=300

        sin_cos = 0
        if (self.settings['method'] == 'EM_VMM') or (self.settings['method'] == 'EM_VMM_soft'): 
            cluster_means = self.cluster_details['EM_VMM_means']
        elif self.settings['method'] == 'k_means':
            if self.settings['sin_cos'] == 1:
                cluster_means = self.cluster_details['k_means_centroids_sc']
                sin_cos = 1
            else:
                cluster_means = self.cluster_details['k_means_centroids']
        elif self.settings['method'] == 'k_means_periodic':
            cluster_means = self.cluster_details['k_means_periodic_centroids']
        elif self.settings['method'] == 'EM_GMM':    
            if self.settings['sin_cos'] == 1:
                cluster_means = self.cluster_details['EM_GMM_means_sc']
                sin_cos = 1
            else:
                cluster_means = self.cluster_details['EM_GMM_means']
        if sin_cos:
            instance_array = np.zeros((self.feature_obj.instance_array.shape[0],self.feature_obj.instance_array.shape[1]*2),dtype=float)
            instance_array[:,::2]=np.cos(self.feature_obj.instance_array)
            instance_array[:,1::2]=np.sin(self.feature_obj.instance_array)
            pass
        else:
            #instance_array = (self.feature_obj.instance_array)%(2.*np.pi)
            #instance_array[instance_array>np.pi]-=(2.*np.pi)
            instance_array = modtwopi(self.feature_obj.instance_array, offset = 0)
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        # if kappa_cutoff!=None:
        #     averages = np.average(self.cluster_details["EM_VMM_kappas"],axis=1)
        #     cluster_list = np.arange(len(averages))[averages>kappa_cutoff]
        # else:
        #     cluster_list = list(set(self.cluster_assignments))
        cluster_list = list(set(self.cluster_assignments))
        colours = ['r','k','b','y','0.6','0.3','r','k','b','y','0.6','0.3']
        colours.extend(colours)
        marker = ['o','o','o','o','o','o','x','x','x','x','x','x']
        marker.extend(marker)
        if compare_dimensions ==None:
            dims = []

            n_dimensions = self.feature_obj.instance_array.shape[1]
            print 'n_dimensions',n_dimensions
            for dim in range(n_dimensions-1):
                dims.append([dim, dim+1])
        else:
            dims = compare_dimensions
        

        n_dimensions = instance_array.shape[1]
        fig_kh, ax_kh = make_grid_subplots(len(dims), sharex = True, sharey = True)
        if pub_fig:
            fig_kh.set_figwidth(8.48*cm_to_inch)
            fig_kh.set_figheight(8.48*0.8*cm_to_inch)
        for ax_loc,(dim1,dim2) in enumerate(dims):
            print dim1, dim2
            dim1=dim1*combine_factor; dim2=dim2*combine_factor
            print 'b', dim1, dim2, combine_factor
            print dim1,dim1+combine_factor+1
            counter = 0
            for i,cluster in enumerate(cluster_list):
                if np.average(self.cluster_details["EM_VMM_kappas"][i,:])>kappa_ave_cutoff:
                    current_items = self.cluster_assignments==cluster
                    datapoints = instance_array[current_items,:]
                    print datapoints[::decimation,dim1:dim1+combine_factor+1].shape, np.sum(datapoints[::decimation,dim1:dim1+combine_factor+1],axis=1).shape
                    ax_kh[ax_loc].scatter(np.sum(datapoints[::decimation,dim1:dim1+combine_factor],axis=1), np.sum(datapoints[::decimation,dim2:dim2+combine_factor],axis=1),c=colours[counter],marker=marker[counter], alpha=alpha,rasterized=True, edgecolors=colours[counter])
                    if plot_means: ax_kh[ax_loc].plot(cluster_means[i,dim1],cluster_means[i,dim2],colours[i]+marker[i],markersize=8)
                    counter+=1
            ax_kh[ax_loc].text(xlabel_loc,ylabel_loc,r'$\Delta \psi_{}$, vs $\Delta \psi_{}$'.format(dim2/combine_factor+1,dim1/combine_factor+1), horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white',alpha=0.5))
        print('number plotted:{}'.format(counter))
        fig_kh.text(0.5, 0.01, r'$\Delta \psi$', ha='center', va='center', fontsize = 9)
        fig_kh.text(0.01, 0.5, r'$\Delta \psi$', ha='center', va='center', rotation='vertical', fontsize=9)
        ax_kh[-1].set_xlim([-np.pi,np.pi])
        ax_kh[-1].set_ylim([-np.pi,np.pi])
        fig_kh.subplots_adjust(hspace=0.015, wspace=0.015,left=0.10, bottom=0.10,top=0.95, right=0.95)
        #fig_kh.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig_kh.tight_layout()
        if pub_fig:
            fig_kh.savefig(filename, bbox_inches='tight', pad_inches=0.01)

        fig_kh.suptitle(suptitle.replace('_','-'),fontsize=8)
        fig_kh.canvas.draw(); fig_kh.show()
        return fig_kh, ax_kh

    def plot_clusters_phase_lines(self,decimation=1, single_plot = False, kappa_cutoff = None, cumul_sum = False, cluster_list = None, ax = None, colours = None, use_instance_array = True):
        '''Plot all the phase lines for the clusters
        Good clusters will show up as dense areas of line

        SH: 9May2013
        '''
        ax_supplied = False if ax==None else True
        cluster_list_tmp = list(set(self.cluster_assignments))
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        n_clusters = len(cluster_list_tmp)
        if cluster_list==None:  cluster_list = cluster_list_tmp
        if single_plot:
            if not ax_supplied: fig, ax = pt.subplots(); ax = [ax]*n_clusters
            if colours == None: colours = ['r','k','b','y','m']*10
        else:
            if not ax_supplied: fig, ax = make_grid_subplots(n_clusters, sharex = True, sharey = True)
            if colours == None: colours = ['k']*n_clusters
        if kappa_cutoff!=None:
            averages = np.average(self.cluster_details["EM_VMM_kappas"],axis=1)
            cluster_list = np.arange(len(averages))[averages>kappa_cutoff]
        means = []
        count = 0
        axes_list = []
        for cluster in cluster_list:
            current_items = self.cluster_assignments==cluster
            if np.sum(current_items)>10:
                if use_instance_array:
                    tmp = modtwopi(self.feature_obj.instance_array[current_items,:], offset = 0)
                else:
                    tmp = np.angle(self.feature_obj.instance_array_amps[current_items,:]/(np.sum(self.feature_obj.instance_array_amps[current_items,:],axis = 1)[:,np.newaxis]))
                    #tmp = modtwopi(tmp, offset = 0)
                    
                #instance_array_complex = np.exp(1j*instance_array)
                #kappa_list_tmp, cur_mean, scale_fit1 = EM_VMM_calc_best_fit(np.exp(1j*self.feature_obj.instance_array[current_items,:]), lookup=None)
                kappa_list_tmp, cur_mean, scale_fit1 = EM_VMM_calc_best_fit(np.exp(1j*tmp), lookup=None)
                cur_mean = modtwopi(cur_mean, offset = 0)
                #cur_mean = modtwopi(self.cluster_details['EM_VMM_means'][cluster,:], offset = 0)
                means.append(cur_mean)
                if cumul_sum:
                    tmp = np.cumsum(tmp,axis = 1)/(2.*np.pi)
                    means[-1] = np.cumsum(means[-1])/(2.*np.pi)
                plot_ax = ax[cluster] if not ax_supplied else ax[count]
                tmp1 = tmp[::decimation,:]
                while np.sum((tmp1 - cur_mean[np.newaxis,:]) > np.pi)>0:
                    tmp1[(tmp1 - cur_mean[np.newaxis,:]) > np.pi]-=2.*np.pi
                while np.sum((tmp1 - cur_mean[np.newaxis,:]) < -np.pi)>0:
                    tmp1[(tmp1 - cur_mean[np.newaxis,:]) < -np.pi]+=2.*np.pi
                plot_ax.plot(tmp1.T, linestyle = '-',linewidth=0.05, color = colours[count], zorder = 0)
                axes_list.append(plot_ax)
                count+=1
                #ax[cluster].legend(loc='best')
        for mean, color, clust, axis in zip(means,colours, cluster_list, axes_list):
            axis.plot(mean,linestyle = '-',linewidth=1.5, color = 'k')
            axis.plot(mean,linestyle = '-',linewidth=1.0, color = 'r')
            #ax[0].set_xlim([0,self.cluster_details['EM_VMM_means'].shape[1]])
            ax[0].set_xlim([0,self.feature_obj.instance_array.shape[1]])
            if not cumul_sum: ax[0].set_ylim([-np.pi, np.pi])
            for i in ax:i.grid(True)
        if not ax_supplied:
            fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
            fig.suptitle(suptitle.replace('_','-'), fontsize = 8)
            fig.canvas.draw(); fig.show()
            return fig, ax

    def plot_clusters_polarisations(self, coil_numbers = 0, cluster_list = None, ax = None, colours = None, scatter_kwargs = None, decimate = 1, polar_plot = False, y_axis = None, reference_phase = 'b_par', plot_circle = True, plot_center = True, plot_text = True, add_perp = True, noise_amp = 0.75, data_key = 'naked_coil', pub_fig = False, fig_name = None, inc_title = False, energy = False, plot_amps = False, plot_distance = False, angle_error = 10):
        '''
        Plot the cluster polarisations
        coil_numbers : list of probe formers to plot
        cluster_list: list of clusters to plot
        ax : supplied axes
        decimate
        polar_plot : whether to make a polar plot of the phases or not
        y_axis: if none it will be versus probe number with some scatter added
        reference_phase : b_par, b_perp_perp, b_perp_surf
        noise_amp = amplitude of noise added for scatter if y_axis == None
        add_perp: add the two perpendicular components to get a total perpendicular amplitude
        plot_distance: plot distance to LCFS for the probes
        energy [True, False]: plot B**2

        SH: 28July2014
        '''
        if fig_name == None: fig_name = 'polarisation'
        if scatter_kwargs == None: scatter_kwargs = {'s':100, 'alpha':0.05,'linewidth':'1'}
        ax_supplied = False if ax==None else True
        cluster_list_tmp = list(set(self.cluster_assignments))
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        if cluster_list==None:  cluster_list = cluster_list_tmp
        n_clusters = len(cluster_list)
        if not ax_supplied: fig, ax, ax_unflat = make_grid_subplots(n_clusters, sharex = True, sharey = True, return_unflattened = True)

        #Divide by two for +-
        angle_error = np.deg2rad(angle_error)/2.
        if len(np.array(ax_unflat).shape)==1:
            ax_unflat = np.array(ax_unflat)[np.newaxis,:]
        else:
            ax_unflat = np.array(ax_unflat)
        import h1.helper.generic_funcs as gen_funcs
        if pub_fig: gen_funcs.setup_publication_image(fig, height_prop = 0.2+0.8*(ax_unflat.shape[0]/2.), single_col = True, fig_width = None, replacement_kwargs = None, fig_height = None)

        if colours == None: colours = ['k']*n_clusters
        count = 0
        #Get the eq mag field vector components if they haven't been obtained already
        #This does not take the different configurations into account....
        if not hasattr(self,'hma'):
            import h1.diagnostics.magnetics as mag
            self.hma = mag.HMA()
            #Why is the hard coded in here? SRH:6July2015
            #kh = 0.33
            #HOME = os.environ['HOME']
            #filename = HOME + '/code/python/h1_eq_generation/results_jason6/kh{:.3f}-kv1.000fixed/boozmn_wout_kh{:.3f}-kv1.000fixed.nc'.format(kh, kh)
            #self.hma.loc_boozer_coords(filename = filename)
            #self.hma.B_unit_vectors(filename = filename)

        def polar_calc(current_items, coil_number, project = True):
            #current_items = self.cluster_assignments==cluster
            print coil_number*3, coil_number*3+3
            tmp = self.feature_obj.misc_data_dict[data_key][current_items,coil_number*3:coil_number*3+3]
            #tmp[:,2] = -1*tmp[:,2]
            #tmp_shape = tmp.shape
            #tmp = np.random.rand(tmp.shape[0]*tmp.shape[1]) - 0.5 + 1j*(np.random.rand(tmp.shape[0]*tmp.shape[1]) - 0.5)
            #tmp.resize(tmp_shape)
            tmp2 = self.feature_obj.misc_data_dict['freq'][current_items]
            #Should modify they frequency response here
            freq_mods = tmp*0
            #Different set of frequency responses depending on if it is the naked coil or in the bellows
            if coil_number!=0:
                freq_mod_f = [self.hma.hma_transverse_bellows, self.hma.hma_axial_bellows, self.hma.hma_transverse_bellows]
            else:
                freq_mod_f = [self.hma.hma_no_bellows, self.hma.hma_no_bellows, self.hma.hma_no_bellows,]
            #make an array the same size as the probe data to frequency correct
            for i in range(3):freq_mods[:,i] = freq_mod_f[i](tmp2)
            print '####', coil_number, cluster
            print np.mean(np.abs(freq_mods), axis = 0), np.mean(tmp2), np.std(tmp2)
            print np.rad2deg(np.angle(np.sum(freq_mods, axis = 0)))
            
            tmp /= freq_mods
            #normalise the vector - need to be careful because they are complex
            norms = (np.linalg.norm(tmp, axis = 1))
            tmp /= norms[:,np.newaxis]
            self.norms = norms
            #print 'blue, black, grey coil_coords', np.mean(np.abs(tmp),axis = 0)
            #Move to cartesian coords
            tmp_cart = tmp*0
            #The following coordinate transformations can probably be changed into simpler dot products between matrices...
            for ii in range(3):tmp_cart[:,ii] = np.sum(tmp*(self.hma.orientations[coil_number,ii::3])[np.newaxis,:],axis = 1)
            #print 'blue, black, grey cart_coords ', np.mean(np.abs(tmp_cart),axis = 0)
            if project:
                self.b_par = np.sum(tmp_cart * (self.hma.b_hat_par[coil_number,:])[np.newaxis,:], axis = 1)
                self.b_perp_surf = np.sum(tmp_cart * (self.hma.b_hat_perp_in_surf[coil_number,:])[np.newaxis,:], axis = 1)
                self.b_perp_perp = np.sum(tmp_cart * (self.hma.b_hat_perp[coil_number,:])[np.newaxis,:], axis = 1)
            else:
                self.b_par = tmp[:,1]
                self.b_perp_surf = tmp[:,2]
                self.b_perp_perp = tmp[:,0]
            #return b_par, b_perp_surf, b_perp_perp
        if coil_numbers == None: coil_numbers = range(0,16,1)
        tmp_ang = np.linspace(0,2.*np.pi,100)
        #amps = np.zeros((len(cluster_list),len(coil_numbers)),dtype = float)
        amps = np.zeros((len(cluster_list_tmp),len(coil_numbers)),dtype = float)
        amps = np.zeros((len(cluster_list_tmp),16),dtype = float)
        #if len(coil_numbers) == 1 :amps = amps[:,np.newaxis]
        hma_dict = {}
        error_bar_data ={}
        error_bar_data[count] = {}
        for count in range(len(cluster_list)):
            error_bar_data[count] = {}
            for tmp_name in ['combined','par','perp_surf','perp_perp']:
                error_bar_data[count][tmp_name] = {'x_val':[],'y_val':[],'low_err':[], 'upper_err':[]}
        for coil_number in coil_numbers:
            count = 0
            for cluster in cluster_list:
                current_items = self.cluster_assignments==cluster
                if np.sum(current_items)>10:
                    #plot_ax = ax[cluster] if not ax_supplied else ax[count]
                    if cluster in hma_dict.keys():
                        self.hma = hma_dict[cluster]
                    else:
                        import h1.diagnostics.magnetics as mag
                        hma_dict[cluster] = mag.HMA()
                        kh = np.mean(self.feature_obj.misc_data_dict['kh'][current_items])
                        print '!!!!!', kh
                        #kh = 0.33
                        avail = np.array([0.33,0.37,0.44,0.63,0.69,0.83])
                        loc = np.argmin(np.abs(kh - avail))
                        HOME = os.environ['HOME']
                        filename  = HOME + '/code/python/h1_eq_generation/results_jason6/kh%.3f-kv1.000fixed/boozmn_wout_kh%.3f-kv1.000fixed.nc'%(avail[loc], avail[loc])
                        #filename = '/home/srh112/code/python/h1_eq_generation/results_jason6/kh{:.3f}-kv1.000fixed/boozmn_wout_kh{:.3f}-kv1.000fixed.nc'.format(kh, kh)
                        hma_dict[cluster].loc_boozer_coords(filename = filename)
                        hma_dict[cluster].B_unit_vectors(filename = filename)
                        self.hma = hma_dict[cluster]
                    plot_ax = ax[count]
                    #self.b_par, self.b_perp_surf, self.b_perp_perp = polar_calc(current_items, coil_number)
                    polar_calc(current_items, coil_number, project = True)
                    amps[cluster,coil_number] = np.sum(self.norms)
                    if y_axis == None:
                        y_ax_data = coil_number + (np.random.rand(self.b_par.shape[0]))*noise_amp - noise_amp/2
                    else:
                        y_ax_data = self.feature_obj.misc_data_dict[y_axis][current_items]
                    if polar_plot:
                        #use a reference for the phase, and make sure to normalize it so it doesn't affect amplitudes
                        ref = getattr(self,reference_phase)
                        ref = ref/np.abs(ref)
                        for comp, clr in zip([self.b_perp_surf, self.b_perp_perp, self.b_par], ['k','b','r']):
                            if np.allclose(getattr(self,reference_phase), comp):
                                tmp_amp = np.std(np.abs(comp))*3
                                #phase_noise = (np.random.rand(self.b_par.shape[0]))*tmp__amp - tmp_amp/2
                                noise_x = (np.random.rand(self.b_par.shape[0]))*tmp_amp - tmp_amp/2
                                noise_y = (np.random.rand(self.b_par.shape[0]))*tmp_amp - tmp_amp/2
                                noise_x = noise_x[::decimate]
                                noise_y = noise_y[::decimate]
                            else:
                                noise_x = 0
                                noise_y = 0
                            tmp3 = (comp)/ref
                            plot_ax.scatter(np.real(tmp3)[::decimate]+ noise_x, np.imag(tmp3)[::decimate]+ noise_y, c=clr, marker='o', cmap=None, norm=None, zorder=0, rasterized=True,alpha = 0.1)
                            center = [np.mean(np.real(tmp3)), np.mean(np.imag(tmp3))]
                            radius = np.sqrt(np.std(np.real(tmp3))**2+np.std(np.imag(tmp3))**2)
                            if plot_circle:plot_ax.plot(center[0] + radius*np.cos(tmp_ang), center[1]+radius*np.sin(tmp_ang), clr)
                            if plot_center:plot_ax.plot(np.mean(np.real(tmp3)), np.mean(np.imag(tmp3)), clr + 'o')
                            if plot_text:plot_ax.text(np.mean(np.real(tmp3)), np.mean(np.imag(tmp3)), '{}'.format(coil_number))
                    else:
                        tmp_vals = np.abs(self.b_par)[::decimate]
                        if energy:tmp_vals = tmp_vals **2
                        plot_ax.scatter(tmp_vals, y_ax_data[::decimate], c='r', marker='o', cmap=None, norm=None, zorder=0, rasterized=True,alpha = 0.05)
                        #x_err = np.std(tmp_vals)
                        #plot_ax.errorbar(np.mean(tmp_vals), np.mean(y_ax_data[::decimate]),xerr= x_err, fmt = 'x',ecolor='y')
                        ang = np.arccos(np.mean(tmp_vals))
                        #error = np.max(np.abs(np.array([np.cos(ang-angle_error), np.cos(ang+angle_error)]) - np.mean(tmp_vals)))
                        #plot_ax.errorbar([np.mean(tmp_vals)], [np.mean(y_ax_data[::decimate])], xerr=error, ecolor='k', color='k')
                        tmp_ind = error_bar_data[count]['par']
                        tmp_ind['x_val'].append(np.mean(tmp_vals))
                        tmp_ind['y_val'].append(np.mean(y_ax_data[::decimate]))
                        tmp_ind['low_err'].append(np.abs(np.cos(ang-angle_error) - np.mean(tmp_vals)))
                        tmp_ind['upper_err'].append(np.abs(np.cos(ang+angle_error) - np.mean(tmp_vals)))

                        if add_perp:
                            combined = np.linalg.norm(np.array([self.b_perp_surf[::decimate], self.b_perp_perp[::decimate]]).T, axis = 1)
                            if energy:combined = combined **2
                            #plot_ax.scatter(np.linalg.norm(combined, axis = 1), y_ax_data[::decimate], c='b', marker='o', cmap=None, norm=None, zorder=0, rasterized=True,alpha = 0.05)
                            plot_ax.scatter(combined, y_ax_data[::decimate], c='b', marker='o', cmap=None, norm=None, zorder=0, rasterized=True,alpha = 0.05)
                            #x_err = np.std(tmp_vals)
                            #plot_ax.errorbar(np.mean(combined), np.mean(y_ax_data[::decimate]),xerr=x_err, ecolor='y',fmt='x')
                            ang = np.arccos(np.mean(combined))
                            #error = np.max(np.abs(np.array([np.cos(ang-angle_error), np.cos(ang+angle_error)]) - np.mean(combined)))
                            #plot_ax.errorbar([np.mean(combined)], [np.mean(y_ax_data[::decimate])],xerr=error,ecolor='k', color='k')
                            tmp_ind = error_bar_data[count]['combined']
                            tmp_ind['x_val'].append(np.mean(combined))
                            tmp_ind['y_val'].append(np.mean(y_ax_data[::decimate]))
                            tmp_ind['low_err'].append(np.abs(np.cos(ang-angle_error) - np.mean(combined)))
                            tmp_ind['upper_err'].append(np.abs(np.cos(ang+angle_error) - np.mean(combined)))

                        else:
                            tmp_vals = np.abs(self.b_perp_surf)[::decimate]
                            if energy:tmp_vals = tmp_vals **2
                            plot_ax.scatter(tmp_vals, y_ax_data[::decimate], c='b', marker='o', cmap=None, norm=None, zorder=0, rasterized=True,alpha = 0.05)
                            tmp_ind = error_bar_data[count]['perp_surf']
                            tmp_ind['x_val'].append(np.mean(tmp_vals))
                            tmp_ind['y_val'].append(np.mean(y_ax_data[::decimate]))
                            tmp_ind['low_err'].append(np.abs(np.cos(ang-angle_error) - np.mean(tmp_vals)))
                            tmp_ind['upper_err'].append(np.abs(np.cos(ang+angle_error) - np.mean(tmp_vals)))

                            tmp_vals = np.abs(self.b_perp_perp)[::decimate]
                            if energy:tmp_vals = tmp_vals **2
                            plot_ax.scatter(tmp_vals, y_ax_data[::decimate], c='k', marker='o', cmap=None, norm=None, zorder=0, rasterized=True,alpha = 0.05)
                            tmp_ind = error_bar_data[count]['perp_perp']
                            tmp_ind['x_val'].append(np.mean(tmp_vals))
                            tmp_ind['y_val'].append(np.mean(y_ax_data[::decimate]))
                            tmp_ind['low_err'].append(np.abs(np.cos(ang-angle_error) - np.mean(tmp_vals)))
                            tmp_ind['upper_err'].append(np.abs(np.cos(ang+angle_error) - np.mean(tmp_vals)))
                    if inc_title: 
                        if titles==None:
                            plot_ax.set_title('Cluster {}'.format(count + 1))
                        else:
                            plot_ax.set_title('Cluster {}'.format(titles[count]))
                    count+=1
        for count in range(len(ax)):
            plot_ax = ax[count]
            tmp_ind = error_bar_data[count]['par']
            plot_ax.errorbar(tmp_ind['x_val'], tmp_ind['y_val'], xerr=[tmp_ind['upper_err'], tmp_ind['low_err']], fmt='.',ecolor='k', color='k')
            if add_perp:
                tmp_ind = error_bar_data[count]['combined']
                plot_ax.errorbar(tmp_ind['x_val'], tmp_ind['y_val'], xerr=[tmp_ind['upper_err'], tmp_ind['low_err']], fmt='.',ecolor='g', color='g')
            else:
                tmp_ind = error_bar_data[count]['perp_surf']
                plot_ax.errorbar(tmp_ind['x_val'], tmp_ind['y_val'], xerr=[tmp_ind['upper_err'], tmp_ind['low_err']], fmt='.',ecolor='g', color='g')
                tmp_ind = error_bar_data[count]['perp_perp']
                plot_ax.errorbar(tmp_ind['x_val'], tmp_ind['y_val'], xerr=[tmp_ind['upper_err'], tmp_ind['low_err']], fmt='.',ecolor='g', color='g')
        count = 0
        for cluster in cluster_list:
            if plot_amps: ax[count].plot(amps[cluster,:]/np.max(amps[cluster,:]), coil_numbers, 'xb-')
            
            if plot_distance: 
                vals = (self.hma.distance - np.min(self.hma.distance)) * 1./(np.max(self.hma.distance) - np.min(self.hma.distance))
                ax[count].plot(1 - vals, coil_numbers, 'ok-')
            count+=1
        ax[0].set_xlim([0,1])
        if y_axis == None:
            ax[0].set_ylim([-1,17])
        else:
            ax[0].set_ylim([0,60000])
        if polar_plot:
            for tmp_amp in np.arange(0,1.25,0.25):
                for i in ax:i.plot(tmp_amp*np.cos(tmp_ang), tmp_amp*np.sin(tmp_ang),'k')
            ax[0].set_xlim([-1,1])
            ax[0].set_ylim([-1,1])
            
        #if not cumul_sum: ax[0].set_ylim([0, 1]); ax[0].set_xlim([0,1])
        for i in ax:i.grid(True)
        if pub_fig:
            xlab, ylab = ['Imag', 'Real'] if polar_plot else ['Normalised Amplitude','Coil Number']
            if energy and not polar_plot: xlab = 'Energy (a.u)'
            for i in ax_unflat[:,0]: i.set_ylabel(ylab)
            for i in ax_unflat[-1,:]: i.set_xlabel(xlab)
            fig.subplots_adjust(hspace=0.3, wspace=0.15,left=0.05, bottom=0.05,top=0.95, right=0.95)
            fig.tight_layout(pad = 0.3)
            for i in ['pdf','svg','eps']:fig.savefig('{}.{}'.format(fig_name, i))
        if not ax_supplied and not pub_fig:
            fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
            fig.suptitle(suptitle.replace('_','-'), fontsize = 8)
            fig.canvas.draw(); fig.show()
            return fig, ax

    
    def plot_clusters_amp_lines(self,decimation=1):
        '''Plot all the phase lines for the clusters
        Good clusters will show up as dense areas of line

        SH: 9May2013
        '''
        cluster_list = list(set(self.cluster_assignments))
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        n_clusters = len(cluster_list)
        fig, ax = make_grid_subplots(n_clusters, sharex = True, sharey = True)
        for i,cluster in enumerate(cluster_list):
            current_items = self.cluster_assignments==cluster
            if np.sum(current_items)>10:
                divisor = np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,0])
                divisor = np.mean(np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,:]), axis = 0)
                divisor = np.argmax(divisor)
                divisor = np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,divisor])
                divisor = np.mean(np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,:]),axis = 1)
                #divisor = np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,0])
                tmp = (np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,:]).T / divisor).T
                #tmp = (np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,1:]) / np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,0:-1]))
                ax[i].plot(tmp[::decimation,:].T,'k-',linewidth=0.05)
                mean = np.mean(tmp[:,:], axis = 0)
                ax[i].plot(mean,linestyle = '-',linewidth=1.5, color = 'b')
                ax[i].plot(mean,linestyle = '-',linewidth=1.0, color = 'r')

        for i in ax:i.grid(True)
        ax[0].set_xlim([0,tmp.shape[1]])
        ax[0].set_ylim([0,2.5])
        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig.suptitle(suptitle.replace('_','-'), fontsize = 8)
        fig.canvas.draw(); fig.show()
        return fig, ax

    def plot_clusters_re_im_lines(self,decimation=1):
        '''Plot all the phase lines for the clusters
        Good clusters will show up as dense areas of line

        SH: 9May2013
        '''
        cluster_list = list(set(self.cluster_assignments))
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        n_clusters = len(cluster_list)
        fig_re, ax_re = make_grid_subplots(n_clusters, sharex = True, sharey = True)
        #fig_im, ax_im = make_grid_subplots(n_clusters, sharex = True, sharey = True)
        data_complex = self.feature_obj.instance_array_amps/np.sum(self.feature_obj.instance_array_amps, axis = 1)[:,np.newaxis]
        for cluster in cluster_list:
            current_items = self.cluster_assignments==cluster
            if np.sum(current_items)>10:
                # divisor = np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,0])
                # divisor = np.mean(np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,:]), axis = 0)
                # divisor = np.argmax(divisor)
                # divisor = np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,divisor])
                # divisor = np.mean(np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,:]),axis = 1)
                # #divisor = np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,0])
                #data_complex = self.feature_obj.instance_array_amps/(instance_array_amps[:,2])[:,np.newaxis]
                tmp = data_complex[current_items,:]
                #tmp = (np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,:]).T / divisor).T
                #tmp = (np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,1:]) / np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,0:-1]))
                ax_re[cluster].plot(np.real(tmp[::decimation,:].T),linewidth=0.05, color = 'k', linestyle = '-')
                ax_re[cluster].plot(np.imag(tmp[::decimation,:].T),linewidth=0.05, color = 'b', linestyle = '-')
                mean = np.mean(np.real(tmp[:,:]), axis = 0)
                ax_re[cluster].plot(mean,linestyle = '-',linewidth=1.5, color = 'r')
                mean = np.mean(np.imag(tmp[:,:]), axis = 0)
                ax_re[cluster].plot(mean,linestyle = '-',linewidth=1.5, color = 'r')

        for i in ax_re:i.grid(True)
        #for i in ax_im:i.grid(True)
        ax_re[0].set_xlim([0,tmp.shape[1]])
        ax_re[0].set_ylim([-0.75,0.75])
        #ax_im[0].set_xlim([0,tmp.shape[1]])
        #ax_im[0].set_ylim([-0.5,0.5])
        fig_re.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        #fig_im.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig_re.suptitle(suptitle.replace('_','-'), fontsize = 8)
        fig_re.canvas.draw(); fig_re.show()
        #fig_im.suptitle(suptitle.replace('_','-'), fontsize = 8)
        #fig_im.canvas.draw(); fig_re.show()
        #return fig, ax

    def plot_fft_amp_lines(self,decimation=1):
        '''Plot all the phase lines for the clusters
        Good clusters will show up as dense areas of line

        SH: 9May2013
        '''
        cluster_list = list(set(self.cluster_assignments))
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        n_clusters = len(cluster_list)
        fig, ax = make_grid_subplots(n_clusters, sharex = True, sharey = True)
        for cluster in cluster_list:
            current_items = self.cluster_assignments==cluster
            if np.sum(current_items)>10:
                #tmp = (np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,:]).T / np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,0])).T
                tmp = np.fft.fft(self.feature_obj.misc_data_dict['mirnov_data'][current_items,:])
                print np.max(np.abs(tmp),axis=1).shape
                tmp = ((np.abs(tmp).T)/np.max(np.abs(tmp),axis=1)).T
                ax[cluster].plot(np.abs(tmp[::decimation,:]).T,'k-',linewidth=0.05)
        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig.suptitle(suptitle, fontsize = 8)
        fig.canvas.draw(); fig.show()
        return fig, ax

    def plot_interferometer_channels(self, interferometer_spacing=0.025, interferometer_start=0,  include_both_sides = 1, plot_phases=0):
        '''plot kh vs frequency for each cluster - i.e looking for whale tails
        The colouring of the points is based on the total phase along the array
        i.e a 1D indication of the clusters

        SH: 9May2013
        '''
        cluster_list = list(set(self.cluster_assignments))
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        n_clusters = len(cluster_list)
        fig, ax = make_grid_subplots(n_clusters, sharex = True, sharey = True)
        for i in ax: i.set_rasterization_zorder(1)
        if plot_phases : fig_phase, ax_phase = make_grid_subplots(n_clusters, sharex = True, sharey = True)
        passed = True; i=1; ne_omega_data = []
        misc_data_dict = self.feature_obj.misc_data_dict
        #strange way to determine the number of channels we have.... need to figureout a better way
        ne_omega_data = misc_data_dict['ne_mode']

        #this is a bad fudge....
        #channel_list = np.arange(interferometer_start,interferometer_spacing*len(ne_omega_data),interferometer_spacing)
        channel_list = np.arange(interferometer_start,interferometer_spacing*ne_omega_data.shape[1],interferometer_spacing)
        #ne_omega_data = np.array(ne_omega_data).transpose()
        for cluster in cluster_list:
            current_items = self.cluster_assignments==cluster
            if np.sum(current_items)>10:
                current_data = np.abs(ne_omega_data[current_items,:])
                rms_amp = np.sqrt(np.sum((current_data * current_data),axis=1))
                mean_rms = np.mean(rms_amp)
                large_enough = rms_amp >mean_rms
                current_data = current_data / np.tile(rms_amp,(7,1)).transpose()
                if plot_phases:  
                    current_phase = np.angle(ne_omega_data[current_items,:])
                    current_phase = current_phase - np.tile(current_phase[:,2],(7,1)).transpose()
                    current_phase = np.rad2deg(current_phase %(2.*np.pi))
                ax[cluster].plot(channel_list, current_data[large_enough].T,'k-',linewidth=0.02,zorder = 0)
                means = np.mean(current_data[large_enough], axis=0)
                std = np.std(current_data[large_enough], axis=0)
                ax[cluster].errorbar(channel_list, means, yerr=std)
                ax[cluster].plot(channel_list, means,'c-o',linewidth=6)
                ax[cluster].plot(channel_list, means,'b-o',linewidth=4)
                ax[cluster].plot(channel_list*-1, means,'c--o',linewidth=6)
                ax[cluster].plot(channel_list*-1, means,'b--o',linewidth=4)
                ax[cluster].grid()
        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        if plot_phases:
            fig_phase.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
            fig_phase.suptitle(suptitle,fontsize=8)
            fig_phase.canvas.draw(); fig_phase.show()
        fig.suptitle(suptitle,fontsize=8)
        fig.canvas.draw(); fig.show()
        return fig, ax

    def plot_single_kh(self, cluster_list = None,kappa_cutoff=None,color_by_cumul_phase = 1, sqrtne=None, plot_alfven_lines=1,xlim=None,ylim=None,pub_fig = 0, filename = None, marker_size = 100, alpha = 0.05, linewidth='1', colour_list = None):
        '''plot kh vs frequency for each cluster - i.e looking for whale tails
        The colouring of the points is based on the total phase along the array
        i.e a 1D indication of the clusters

        can provide a cluster_list to select which ones are plotted
        or can give a kappa_cutoff (takes precedent)
        otherwise, all will be plotted
        SH: 9May2013
        '''
        if pub_fig:
            cm_to_inch=0.393701
            import matplotlib as mpl
            old_rc_Params = mpl.rcParams
            mpl.rcParams['font.size']=8.0
            mpl.rcParams['axes.titlesize']=8.0#'medium'
            mpl.rcParams['xtick.labelsize']=8.0
            mpl.rcParams['ytick.labelsize']=8.0
            mpl.rcParams['lines.markersize']=0.5
            mpl.rcParams['savefig.dpi']=300

        misc_data_dict = self.feature_obj.misc_data_dict
        if kappa_cutoff!=None:
            averages = np.average(self.cluster_details["EM_VMM_kappas"],axis=1)
            fig, ax = pt.subplots()
            kappa_cutoff_list = range(50)
            items = []
            for kappa_cutoff_tmp in kappa_cutoff_list:
                cluster_list = np.arange(len(averages))[averages>kappa_cutoff_tmp]
                total = 0
                for i in cluster_list: total+= np.sum(self.cluster_assignments==i)
                items.append(total)
            ax.plot(kappa_cutoff_list, items,'o')
            fig.canvas.draw(); fig.show()
            cluster_list = np.arange(len(averages))[averages>kappa_cutoff]
            total = 0
            for i in cluster_list: total+= np.sum(self.cluster_assignments==i)
            print('total clusters satisfying kappa bar>{}:{}'.format(kappa_cutoff,total))

        elif cluster_list ==None:
            cluster_list = list(set(self.cluster_assignments))
        kh_plot_item = 'kh'; freq_plot_item = 'freq'
        fig, ax = pt.subplots(); ax = [ax]
        if pub_fig:
            fig.set_figwidth(8.48*cm_to_inch)
            fig.set_figheight(8.48*0.8*cm_to_inch)
        for i in ax: i.set_rasterization_zorder(1)
        if color_by_cumul_phase:
            instance_array2 = modtwopi(self.feature_obj.instance_array, offset=0)
            max_lim = -1*2.*np.pi; min_lim = (-3.*2.*np.pi)
            total_phase = np.sum(instance_array2,axis=1)
            total_phase = np.clip(total_phase,min_lim, max_lim)
        plotting_offset = 0
        if colour_list==None: colour_list = ['k','b','r','y','g','c','m','w']
        marker_list = ['o' for i in colour_list]
        marker_list.extend(['d' for i in colour_list])
        colour_list.extend(colour_list)
        print 'hello'
        while len(colour_list)< len(cluster_list):
            print 'hello'
            colour_list.extend(colour_list)
            marker_list.extend(marker_list)
        print cluster_list
        for i,cluster in enumerate(cluster_list):
            current_items = self.cluster_assignments==cluster
            if sqrtne==None:
                scatter_data = misc_data_dict[freq_plot_item][current_items]/1000
            else:
                scatter_data = misc_data_dict[freq_plot_item][current_items]*np.sqrt(misc_data_dict['ne{ne}'.format(ne=sqrtne)][current_items])
                print 'scaling by ne'
            if np.sum(current_items)>10:
                if color_by_cumul_phase == 1:
                    print 'hello instance', cluster
                    ax[0].scatter((misc_data_dict[kh_plot_item][current_items]), scatter_data, s=marker_size, c=total_phase[current_items], vmin = min_lim, vmax = max_lim, marker='o', cmap='jet', norm=None, alpha=alpha, linewidth=linewidth)
                elif color_by_cumul_phase ==0:
                    ax[0].scatter((misc_data_dict[kh_plot_item][current_items]), scatter_data,s=marker_size, c=colour_list[i], marker=marker_list[i], cmap=None, norm=None, alpha=alpha,zorder=0,rasterized=True, linewidth=linewidth)
                    print 'hello, no instance', cluster
                elif color_by_cumul_phase == 2:
                    ax[0].scatter((misc_data_dict[kh_plot_item][current_items]), scatter_data,s=marker_size, c='k', marker='o', cmap=None, norm=None, alpha=alpha,zorder=0,rasterized=True, linewidth=linewidth)
                    print 'hello, no instance', cluster
        if plot_alfven_lines:
            #plot_alfven_lines_func(ax[0])
            print("Commented out plot_alfven_lines because we have no theoretical lines yet... " )
        ax[-1].set_xlim([0.201,0.99])
        ax[0].set_xlabel(r'$\kappa_H$')
        ax[0].grid()
        #ax[0].set_ylabel(r'$\omega \sqrt{n_e}$')
        ax[0].set_ylabel('Frequency (kHz)')
        if xlim!=None: ax[0].set_xlim(xlim)
        if ylim!=None: ax[0].set_ylim(ylim)
        if pub_fig:
            fig.savefig(filename, bbox_inches='tight', pad_inches=0.01)

        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig.canvas.draw(); fig.show()
        return fig, ax

    def plot_cumulative_phases(self,):
        if ((self.settings['method']) == ('EM_VMM')) or ((self.settings['method']) == ('EM_VMM_soft')): 
            means = self.cluster_details['EM_VMM_means']
        elif self.settings['method'] == 'k_means':
            means = self.cluster_details['k_means_centroids']
        elif self.settings['method'] == 'k_means_periodic':
            means = self.cluster_details['k_means_periodic_centroids']

        cluster_list = list(set(self.cluster_assignments))
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        n_clusters = len(cluster_list)
        fig, ax = make_grid_subplots(n_clusters, sharex = True, sharey = True)
        for cluster in cluster_list:
            cluster_phases = means[cluster][:]
            cumulative_phase = [0]
            for tmp_angle in range(len(cluster_phases)):
                cumulative_phase.append(cumulative_phase[-1] + cluster_phases[tmp_angle])
            cumulative_phase = np.array(cumulative_phase)/(2.*np.pi)
            ax[cluster].plot(range(len(cumulative_phase)),cumulative_phase,'o-')
            colors = ['k','b','r']; plot_style = ['-o','-x','-s']; plot_style2 = ['--o','--x','--s']

            #approximate coil locations in Boozer Coordinates
            min_locations_theta = np.array([297.165,268.283,239.198,211.107,185.257,160.934,
                                            137.809,114.798,92.381,70.123,46.695,21.438,-5.049,
                                            -32.694,-61.366,-90.180])
            min_locations_phi = np.array([46.742,37.825,28.680,19.329,9.922,0.604,-8.267,-16.854,
                                          -24.904,-32.505,-40.037,-47.792,-55.819,-64.136,
                                          -72.807,-81.693])
            #Boozer mode list
            m_mode_list = [3,4,5]; n_mode_list = [-4,-5,-6]
            for j in range(0,len(m_mode_list)):
                m_mode = m_mode_list[j]; n_mode = n_mode_list[j]
                phases = (m_mode * min_locations_theta[1:]/180.*np.pi + n_mode * min_locations_phi[1:]/180.*np.pi)
                diff_phases = np.diff(phases)
                min_amount = -1.5
                diff_phases[diff_phases<min_amount*np.pi]+=2.*np.pi
                diff_phases[diff_phases>(2+min_amount)*np.pi]-=2.*np.pi
                cumulative_phases = (np.cumsum(diff_phases))/2./np.pi
                cumulative_phases = np.append([0],cumulative_phases)
                ax[cluster].plot(cumulative_phases,colors[j],label='(%d,%d)'%(n_mode,m_mode))

        ax[-1].set_ylim([-3,0])
        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig.suptitle(suptitle,fontsize=8)
        fig.canvas.draw(); fig.show()

    def plot_EMM_GMM_amps(self,suptitle = ''):
        fig, ax = make_grid_subplots(self.settings['n_clusters'])
        means = self.cluster_details['EM_GMM_means']
        stds = self.cluster_details['EM_GMM_std']
        for i in range(means.shape[0]):
            ax[i].plot(means[i,:])
            ax[i].plot(means[i,:] + stds[i,:])
            ax[i].plot(means[i,:] - stds[i,:])
        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig.suptitle(suptitle,fontsize=8)
        fig.canvas.draw(); fig.show()


    def cluster_probabilities(self,):
        tmp = np.max(self.cluster_details['zij'],axis=1)
        tmp1 = np.sum(self.cluster_details['zij'],axis=1)
        print 'best prob: {best_prob:.3f}, worst_prob: {worst_prob:.3f}, max row sum: {max_row:.2f}, min row sum: {min_row:.2f}'.format(best_prob = np.max(tmp), worst_prob=np.min(tmp), max_row=np.max(tmp1), min_row=np.min(tmp1))
        n_clusters = len(list(set(self.cluster_assignments)))
        fig, ax = make_grid_subplots(n_clusters, sharex = True, sharey = True)
        for i in list(set(self.cluster_assignments)):
            curr_probs = tmp[self.cluster_assignments==i]
            print 'cluster {clust}, min prob {min:.2f}, max prob {max:.2f}, mean prob {mean:.2f}, std dev {std:.2f}'.format(clust = i, min = np.min(curr_probs), max = np.max(curr_probs), mean = np.mean(curr_probs), std= np.std(curr_probs))
            ax[i].hist(curr_probs, bins=300)
        fig.canvas.draw(); fig.show()

class clusterer_wrapper(clustering_object):
    '''Wrapper around the EM_GMM_clustering function
    Decided to use a wrapper so that it can be used outside of this architecture if needed

    method : k-means, EMM_GMM, k_means_periodic, EM_VMM
    pass settings as kwargs: these are the default settings:
    'k_means': {'n_clusters':9, 'sin_cos':1, 'number_of_starts':30,'seed':None,'use_scikit':1}
    'EM_GMM' : {'n_clusters':9, 'sin_cos':1, 'number_of_starts':30},
    'k_means_periodic' : {'n_clusters':9, 'number_of_starts':10, 'n_cpus':1, 'distance_calc':'euclidean','convergence_diff_cutoff': 0.2, 'iterations': 40, 'decimal_roundoff':2},
    'EM_VMM' : {'n_clusters':9, 'n_iterations':20, 'n_cpus':1, 'start':'k_means',
                'kappa_calc':'approx','hard_assignments':0,'kappa_converged':0.2,
                'mu_converged':0.01,'LL_converged':1.e-4,'min_iterations':10,'verbose':1, 'seeds':None}}
    'EM_GMM2' : {'n_clusters':9, 'n_iterations':20, 'n_cpus':1, 'start':'k_means',
                'kappa_calc':'approx','hard_assignments':0,'kappa_converged':0.2,
                'mu_converged':0.01,'LL_converged':1.e-4,'min_iterations':10,'verbose':1}}

    SH: 6May2013
    '''
    def __init__(self, feature_obj, method='k_means',comment='',amplitude = False, **kwargs):
        self.feature_obj = feature_obj
        #print 'kwargs', kwargs
        #Default settings are declared first, which are overwritten by kwargs
        #if appropriate- this is for record keeping of all settings that are used
        cluster_funcs = {'k_means': k_means_clustering, 'EM_GMM' : EM_GMM_clustering,
                         'k_means_periodic' : k_means_periodic, 'EM_VMM' : EM_VMM_clustering_wrapper,
                         'EM_GMM2' : EM_GMM_clustering_wrapper,
                         'EM_VMM_GMM': EM_VMM_GMM_clustering_wrapper,
                         'EM_GMM_GMM': EM_GMM_GMM_clustering,
                         'EM_GMM_GMM2':EM_GMM_GMM2_clustering_wrapper}

        #EM_VMM_clustering,'EM_VMM_soft':EM_VMM_clustering_soft}
        cluster_func_class = {'k_means': 'func', 'EM_GMM' : 'func',
                              'k_means_periodic' : 'func', 'EM_VMM' : 'func', 'EM_GMM2':'func', 
                              'EM_VMM_GMM':'func', 'EM_GMM_GMM':'func', 'EM_GMM_GMM2':'func'}
        
        default_settings = {'k_means': {'n_clusters':9, 'sin_cos':1, 'number_of_starts':30,'seed':None,'use_scikit':1},
                            'EM_GMM' : {'n_clusters':9, 'sin_cos':1, 'number_of_starts':30},
                            'k_means_periodic' : {'n_clusters':9, 'number_of_starts':10, 'n_cpus':1, 'distance_calc':'euclidean','convergence_diff_cutoff': 0.2, 'n_iterations': 40, 'decimal_roundoff':2},
                            'EM_VMM' : {'n_clusters':9, 'n_iterations':20, 'n_cpus':1, 'start':'k_means',
                                        'kappa_calc':'approx','hard_assignments':0,'kappa_converged':0.2,
                                        'mu_converged':0.01,'LL_converged':1.e-4,'min_iterations':10,'verbose':1,'number_of_starts':1, 'seeds':None},
                            'EM_GMM2' : {'n_clusters':9, 'n_iterations':20, 'n_cpus':1, 'start':'k_means',
                                         'kappa_calc':'approx','hard_assignments':0,'kappa_converged':0.2,
                                         'mu_converged':0.01,'LL_converged':1.e-4,'min_iterations':10,'verbose':1,'number_of_starts':1},
                            'EM_VMM_GMM' : {'n_clusters':9, 'n_iterations':20, 'n_cpus':1, 'start':'random',
                                        'kappa_calc':'approx','hard_assignments':0,'kappa_converged':0.2,
                                            'mu_converged':0.01,'LL_converged':1.e-4,'min_iterations':10,'verbose':1,'number_of_starts':1},
                            'EM_GMM_GMM' : {'n_clusters':9, 'number_of_starts':1, 'covariance_type':'diag'},
                            'EM_GMM_GMM2' : {'n_clusters':9, 'n_iterations':20, 'n_cpus':1, 'start':'random',
                                            'kappa_calc':'approx','hard_assignments':0,'kappa_converged':0.2,
                                            'mu_converged':0.01,'LL_converged':1.e-4,'min_iterations':10,'verbose':1,'number_of_starts':1}}

#EM_GMM_GMM_clustering(instance_array_amps, n_clusters=9, sin_cos = 0, number_of_starts = 10, show_covariances = 0, clim=None, covariance_type='diag')
        #EM_VMM_GMM_clustering_wrapper(instance_array, instance_array_amps, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random', kappa_calc='approx', hard_assignments = 0, kappa_converged = 0.1, mu_converged = 0.01, min_iterations=10, LL_converged = 1.e-4, verbose = 0, number_of_starts = 1):
        #replace EM_VMM and EM_VMM_soft with the class.... somehow
        self.settings = default_settings[method]
        self.settings.update(kwargs)
        import jtools as jt
        print("Clustering Settings. . .")
        jt.print_dict(self.settings, "   ")
        print("\n")
        cluster_func = cluster_funcs[method]
        #print method, self.settings
        if amplitude is False:
            print("Amplitude is False")
            input_array = self.feature_obj.instance_array
        else:
            print("Amplitude is True")
            input_array = np.abs(self.feature_obj.misc_data_dict["mirnov_data"])

        if cluster_func_class[method]=='func':
            #print 'func based...'
            if method=='EM_VMM_GMM':
                self.cluster_assignments, self.cluster_details = cluster_func(input_array,
                                                                              self.feature_obj.misc_data_dict[
                                                                                  'mirnov_data'], **self.settings)
            if method=='EM_GMM_GMM2':
                self.cluster_assignments, self.cluster_details = cluster_func(self.feature_obj.misc_data_dict['mirnov_data'], **self.settings)
            elif method=='EM_GMM_GMM':
                self.cluster_assignments, self.cluster_details = cluster_func(self.feature_obj.instance_array_amps, **self.settings)
                #self.cluster_assignments, self.cluster_details = cluster_func(self.feature_obj.misc_data_dict['mirnov_data'], **self.settings)
            else:
                self.cluster_assignments, self.cluster_details = cluster_func(input_array, **self.settings)
        else:
            #print 'class based...'
            tmp = cluster_func(input_array, **self.settings)
            self.cluster_assignments, self.cluster_details = tmp.cluster_assignments, tmp.cluster_details
        self.settings['method']=method
        self.cluster_details['comments'] = comment
        #self.cluster_details['header']='testing'

def normalise_covariances(cov_mat, geom = True):
    n_i, n_j = cov_mat.shape
    normalised_covariance = cov_mat.copy()
    for i in range(n_i):
        for j in range(n_j):
            if geom:
                normalised_covariance[i, j] = np.abs(cov_mat[i,j])/np.sqrt(cov_mat[i,j]**2 + cov_mat[i,i]**2 + cov_mat[j,j]**2)
            else:
                normalised_covariance[i, j] = np.abs(cov_mat[i,j])/(np.abs(cov_mat[i,j]) + np.abs(cov_mat[i,i]) + np.abs(cov_mat[j,j]))

    return normalised_covariance

def pearson_covariances(cov_mat):
    n_i, n_j = cov_mat.shape
    pearson_covariance = cov_mat.copy()
    for i in range(n_i):
        for j in range(n_j):
            pearson_covariance[i, j] = cov_mat[i,j]/np.sqrt(cov_mat[i,i]*cov_mat[j,j])
    return pearson_covariance

###############################################################
def show_covariances(gmm_covars_tmp, clim=None,individual=None,fig_name=None, cmap = 'jet', pearson=False):
    fig, ax = make_grid_subplots(gmm_covars_tmp.shape[0], sharex = True, sharey = True)
    im = []
    mean_PCC_vals = []
    for i in range(gmm_covars_tmp.shape[0]):
        if pearson:
            cur_covar = np.abs(pearson_covariances(gmm_covars_tmp[i,:,:]))
            dim = gmm_covars_tmp.shape[1]
            mean_PCC_vals.append((np.sum(np.abs(pearson_covariances(gmm_covars_tmp[i,:,:]))) - dim)/(dim*dim - dim))
            print 'mean |PCC| : {}'.format(mean_PCC_vals[-1])
        else:
            cur_covar = np.abs(gmm_covars_tmp[i,:,:])
        im.append(ax[i].imshow(cur_covar,aspect='auto', interpolation='nearest', cmap=cmap))
        print im[-1].get_clim()
        if clim==None:
            im[-1].set_clim([0, im[-1].get_clim()[1]*0.5])
        else:
            im[-1].set_clim(clim)
    if individual!=None:
        fig_ind,ax_ind = pt.subplots(nrows=len(individual),sharex = True,sharey = True)
        if fig_name!=None:
            cm_to_inch=0.393701
            import matplotlib as mpl
            old_rc_Params = mpl.rcParams
            mpl.rcParams['font.size']=8.0
            mpl.rcParams['axes.titlesize']=8.0#'medium'
            mpl.rcParams['xtick.labelsize']=8.0
            mpl.rcParams['ytick.labelsize']=8.0
            mpl.rcParams['lines.markersize']=1.0
            mpl.rcParams['savefig.dpi']=300
            fig_ind.set_figwidth(8.48*cm_to_inch)
            fig_ind.set_figheight(8.48*0.8*cm_to_inch)

        if len(individual)==1:ax_ind = [ax_ind]
        for i,clust in enumerate(individual):
            if pearson:
                cur_covar = np.abs(pearson_covariances(gmm_covars_tmp[clust,:,:]))
                dim = gmm_covars_tmp.shape[1]
                mean_PCC = (np.sum(np.abs(pearson_covariances(gmm_covars_tmp[clust,:,:]))) - dim)/(dim*dim - dim)
                print 'mean |PCC| : {}'.format(mean_PCC)
            else:
                cur_covar = np.abs(gmm_covars_tmp[clust,:,:])
            im = ax_ind[i].imshow(cur_covar,aspect='auto', interpolation='nearest', cmap=cmap)
            cbar = pt.colorbar(im,ax=ax_ind[i])
            if pearson:
                cbar.set_label('| PCC |')
            else:
                cbar.set_label('covariance')
            im.set_clim(clim)
            ax_ind[i].set_ylabel('Dimension')
        ax_ind[-1].set_xlabel('Dimension')
        if fig_name!=None:
            fig_ind.savefig(fig_name, bbox_inches='tight', pad_inches=0.01)
        fig_ind.canvas.draw();fig_ind.show()
    #for i in im : i.set_clim(clims)
    print mean_PCC_vals
    print np.mean(mean_PCC_vals)
    fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
    fig.canvas.draw();fig.show()


##################Clustering wrappers#########################
def EM_GMM_clustering(instance_array, n_clusters=9, sin_cos = 0, number_of_starts = 10, show_covariances = 0, clim=None, covariance_type='diag'):
    print 'starting EM-GMM algorithm from sckit-learn, k=%d, retries : %d, sin_cos = %d'%(n_clusters,number_of_starts,sin_cos)
    if sin_cos==1:
        print '  using sine and cosine of the phases'
        sin_cos_instances = np.zeros((instance_array.shape[0],instance_array.shape[1]*2),dtype=float)
        sin_cos_instances[:,::2]=np.cos(instance_array)
        sin_cos_instances[:,1::2]=np.sin(instance_array)
        input_data = sin_cos_instances
    else:
        print '  using raw phases'
        input_data = instance_array
    gmm = mixture.GMM(n_components=n_clusters,covariance_type=covariance_type,n_init=number_of_starts)
    gmm.fit(input_data)
    cluster_assignments = gmm.predict(input_data)
    bic_value = gmm.bic(input_data)
    LL = np.sum(gmm.score(input_data))
    gmm_covars_tmp = np.array(gmm._get_covars())
    if show_covariances:
        fig, ax = make_grid_subplots(gmm_covars_tmp.shape[0], sharex = True, sharey = True)
        im = []
        for i in range(gmm_covars_tmp.shape[0]):
            im.append(ax[i].imshow(np.abs(gmm_covars_tmp[i,:,:]),aspect='auto'))
            print im[-1].get_clim()
            if clim==None:
                im[-1].set_clim([0, im[-1].get_clim()[1]*0.5])
            else:
                im[-1].set_clim(clim)
        clims = [np.min(np.abs(gmm_covars_tmp)),np.max(np.abs(gmm_covars_tmp))*0.5]
        #for i in im : i.set_clim(clims)
        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig.canvas.draw();fig.show()

    gmm_covars = np.array([np.diagonal(i) for i in gmm._get_covars()])
    gmm_means = gmm.means_
    if sin_cos:
        cluster_details = {'EM_GMM_means_sc':gmm_means, 'EM_GMM_variances_sc':gmm_covars, 'EM_GMM_covariances_sc':gmm_covars_tmp,'BIC':bic_value, 'LL':LL}
    else:
        cluster_details = {'EM_GMM_means':gmm_means, 'EM_GMM_variances':gmm_covars, 'EM_GMM_covariances':gmm_covars_tmp, 'BIC':bic_value,'LL':LL}
    return cluster_assignments, cluster_details

def k_means_clustering(instance_array, n_clusters=9, sin_cos = 1, number_of_starts = 30, seed=None, use_scikit=1,**kwargs):
    '''
    This runs the k-means clustering algorithm as implemented in scipy - change to scikit-learn?

    SH: 7May2013
    '''
    from sklearn.cluster import KMeans
    print 'starting kmeans algorithm, k=%d, retries : %d, sin_cos = %d'%(n_clusters,number_of_starts,sin_cos)
    if sin_cos==1:
        print '  using sine and cosine of the phases'
        sin_cos_instances = np.zeros((instance_array.shape[0],instance_array.shape[1]*2),dtype=float)
        sin_cos_instances[:,::2]=np.cos(instance_array)
        sin_cos_instances[:,1::2]=np.sin(instance_array)
        input_array = sin_cos_instances
        #code_book,distortion = vq.kmeans(sin_cos_instances, n_clusters,iter=number_of_starts)
        #cluster_assignments, point_distances = vq.vq(sin_cos_instances, code_book)
    else:
        print '  using raw phases'
        input_array = instance_array
        #code_book,distortion = vq.kmeans(instance_array, n_clusters,iter=number_of_starts)
        #cluster_assignments, point_distances = vq.vq(instance_array, code_book)
    #pickle.dump(multiple_run_results,file(k_means_output_filename,'w'))
    if use_scikit:
        print 'using scikit learn'
        tmp = KMeans(init='k-means++', n_clusters=n_clusters, n_init = number_of_starts, n_jobs=1, random_state = seed)
        cluster_assignments = tmp.fit_predict(input_array)
        code_book = tmp.cluster_centers_
    else:
        print 'using vq from scipy'
        code_book,distortion = vq.kmeans(input_array, n_clusters,iter=number_of_starts)
        cluster_assignments, point_distances = vq.vq(input_array, code_book)
    if sin_cos:
        cluster_details = {'k_means_centroids_sc':code_book}
    else:
        cluster_details = {'k_means_centroids':code_book}
    return cluster_assignments, cluster_details

##################################################################################
#############################k-means periodic algorithm##############################
def new_method(tmp_array):
    #find the items above and below np.pi as this will determine their wrapping behaviour
    gt_pi = tmp_array>=np.pi
    less_pi = tmp_array<np.pi
    items = tmp_array.shape[0]

    #find the breakpoints for wrapping
    break_points = copy.deepcopy(tmp_array)
    break_points[less_pi] += np.pi
    break_points[gt_pi] -= np.pi

    #create a list of unique subintervals in order and append 2pi to it
    subintervals = np.append(np.unique(break_points), 2.*np.pi)
    wk = []; q = []

    #total sum of the array - this will be modified by factors of 2pi depending on which
    #sub interval we are in
    total_sum = np.sum(tmp_array)
    for i in range(0,subintervals.shape[0]):
        tmp_points2 = (break_points<=subintervals[i])*(less_pi)
        tmp_points3 = (break_points>subintervals[i])*(gt_pi)
        #tmp_array2[tmp_points2] += 2.*np.pi
        
        #calculate new centroid and make sure it lies in <0, 2pi)
        wk.append(((total_sum + (np.sum(tmp_points2)- np.sum(tmp_points3))*2.*np.pi)/items)%(2.*np.pi))
        #old way - not possible mistake with (wk[-1] < 2.*np.pi) - should be <0???
        #wk.append((total_sum + (np.sum(tmp_points2)- np.sum(tmp_points3))*2.*np.pi)/items)
        # while (wk[-1] >= 2.*np.pi) or (wk[-1]< 0):
        #     if (wk[-1] >= 2.*np.pi):
        #         wk[-1] -= 2.*np.pi
        #     elif (wk[-1] < 2.*np.pi):
        #         wk[-1] += 2.*np.pi
        tmp = np.abs(tmp_array - wk[-1])
        tmp = np.minimum(tmp, 2.*np.pi-tmp)
        q.append(0.5*np.sum(tmp**2))
    return wk[np.argmin(q)], np.min(q)

def _k_means_p_centroid_1d(tmp_array):
    '''Function to find the cluster centroid in one dimension
    around a circle

    Note this has become fairly highly vectorised and is the most
    time consuming part of the calcuation

    SH: 9May2013
    '''
    #find the items above and below np.pi as this will determine their wrapping behaviour
    tmp_array = np.sort(tmp_array)
    items = len(tmp_array)
    subintervals= np.unique(tmp_array)
    n_subintervals = len(subintervals)
    subinterval_locs = tmp_array.searchsorted(subintervals)
    items_per_subinterval = np.diff(subinterval_locs)
    items_per_subinterval = np.append(items_per_subinterval, items - np.sum(items_per_subinterval))
    #total_sum = np.sum(tmp_array)
    total_sum = np.sum(items_per_subinterval*subintervals) #marginally faster
    gt_pi_index = subintervals.searchsorted(np.pi)

    #deal with subintervals from 0->pi
    #find the breakpoints for wrapping
    break_points = np.zeros(n_subintervals,dtype=np.float)
    break_points[0:gt_pi_index] = subintervals[0:gt_pi_index]+np.pi
    break_points[gt_pi_index:] = subintervals[gt_pi_index:]-np.pi
    points_to_end = np.cumsum(items_per_subinterval[::-1])
    points_from_start = np.cumsum(items_per_subinterval)

    correction = np.zeros(n_subintervals,dtype=np.float)
    indices = np.minimum(subintervals.searchsorted(break_points[0:gt_pi_index]),n_subintervals-1)
    correction[:gt_pi_index] = points_to_end[indices]*(-2.*np.pi)

    indices = subintervals.searchsorted(break_points[gt_pi_index:])
    correction[gt_pi_index:] = points_from_start[indices]*(2.*np.pi)

    averages = ((total_sum + correction) / items)%(2.*np.pi)

    sub_ints1 = np.tile(subintervals,(n_subintervals,1))
    tmp = np.abs(sub_ints1 - averages[:,np.newaxis])

    tmp = np.minimum(tmp,2.*np.pi-tmp) * items_per_subinterval[:,np.newaxis]
    q_vals = np.sum(tmp**2,axis=1)
    return averages[np.argmin(q_vals)], np.min(q_vals)



def _k_means_p_centroid_1d_complex_average(tmp_array):
    '''Function to find the cluster centroid in one dimension
    around a circle - using the average of the complex numbers...
    this is a biased estimator....

    SH: 9May2013
    '''
    #find the items above and below np.pi as this will determine their wrapping behaviour
    c = np.mean(np.cos(tmp_array))
    s = np.mean(np.sin(tmp_array))
    mean = np.arctan2(s,c)
    distances = np.abs(tmp_array - mean)
    qvals = np.sum((np.minimum(distances, 2.*np.pi-distances))**2)
    return mean, qvals
    #sub_ints1 = np.tile(subintervals,(n_subintervals,1))
    #tmp = np.abs(sub_ints1 - averages[:,np.newaxis])

    # tmp = np.minimum(tmp,2.*np.pi-tmp) * items_per_subinterval[:,np.newaxis]
    # q_vals = np.sum(tmp**2,axis=1)
    # return averages[np.argmin(q_vals)], np.min(q_vals)



def _k_means_p_calc_centroids(centroid, cluster_assignments, instance_array, k):
    '''Find all the new centroids in all dimensions for all clusters
    uses _k_means_p_centroid_1d to perform the calculation

    SH: 9May2013
    '''
    q_tot = 0
    for i in range(0,k):
        relevant_indices = (cluster_assignments == i)
        #ignore clusters without any members
        if instance_array[relevant_indices,:].shape[0] == 0:
            pass
        else:
            #extract the members of the cluster
            tmp_array = instance_array[relevant_indices,:]
            #treat each attribute seperately, and add up the q values
            for j in range(0,tmp_array.shape[1]):
                #centroid[i,j], q_val = _k_means_p_centroid_1d(tmp_array[:,j])
                centroid[i,j], q_val = _k_means_p_centroid_1d_complex_average(tmp_array[:,j])
                q_tot += q_val
    return centroid, q_tot

def _k_means_p_calc_distance(instance_array, centroids, distance_calc = 'euclidean'):
    '''Calculate the distances for all points to all clusters
    this is used to assign the points to clusters 

    SH:9May2013
    '''
    distances = np.ones((instance_array.shape[0],centroids.shape[0]),dtype=np.float)
    for i in range(0, centroids.shape[0]):
        tmp = np.abs(instance_array - np.tile(centroids[i,:],(instance_array.shape[0],1)))
        tmp = np.minimum(tmp, 2.*np.pi-tmp)
        if distance_calc == 'manhatten':
            tmp = np.sum(np.abs(tmp),axis = 1)
        else:
            tmp = np.sum(np.abs(tmp*tmp),axis = 1)
        distances[:,i] = copy.deepcopy(tmp)
    cluster_assignments = np.argmin(distances,axis = 1)
    return cluster_assignments

def _k_means_p_rand_centroids(k, d, seed):
    '''
    Create the random centroid array to start the k-means algorithm 
    based on a seed so that it is repeatable.

    SH: 8May2013
    '''
    np.random.seed(seed)
    centroids = (2.*np.pi)*np.random.random((k,d))#-np.pi
    return centroids

def _k_means_p_single_seed(k, seed, instance_array, distance_calc, convergence_diff_cutoff, iterations):
    '''
    This is the function that calculates the k-means periodic clustering

    SH: 8May2013
    '''
    centroids = _k_means_p_rand_centroids(k, instance_array.shape[1], seed)
    old_centroid = copy.copy(centroids)
    convergence = []; q_list = []
    current_iteration = 0; curr_diff = np.inf
    while (curr_diff > convergence_diff_cutoff) and (current_iteration < iterations):
        current_iteration += 1
        start_time = time.time()
        cluster_assignments = _k_means_p_calc_distance(instance_array, centroids, distance_calc = distance_calc)
        distance_time = time.time()
        new_centroids, q_curr = _k_means_p_calc_centroids(copy.copy(centroids), cluster_assignments, instance_array, k)
        centroid_time = time.time()
        q_list.append(q_curr)
        convergence.append(np.sum(np.abs(new_centroids - centroids)))
        centroids = copy.copy(new_centroids)
        if current_iteration >= 2:
            curr_diff = np.abs(q_list[-1] - q_list[-2])
        print 'pid : %d, iteration : %3d, convergence : %10.3f, q_diff : %10.4f, q_tot : %10.2f, times : %5.3fs %.3fs %.3fs'%(os.getpid(), current_iteration, convergence[-1], curr_diff, q_list[-1], distance_time-start_time, centroid_time-distance_time, time.time() - start_time)
    return centroids, cluster_assignments, q_list[-1]

def _k_means_p_multiproc_wrapper(arguments):
    '''Wrapper around _k_means_p_single_seed so that it can be used with
    multiprocessing, and be passed multiple arguments

    SH: 8May2013
    '''
    print 'started wrapper'
    return _k_means_p_single_seed(*arguments)

def k_means_periodic(instance_array, n_clusters = 9, number_of_starts = 10, n_cpus=1, distance_calc = 'euclidean',convergence_diff_cutoff = 0.2, n_iterations = 40, decimal_roundoff=2,seed_list=None, **kwargs):
    k = n_clusters
    #round, take relevant columns and make sure all datapoints are on the interval <0,2pi)
    instance_array = copy.deepcopy(instance_array)
    instance_array = instance_array.astype(np.float)
    #take modulus 2pi
    instance_array = instance_array % (2.*np.pi)
    #round to decimal roundoff places
    instance_array = np.round(np.array(instance_array),decimals = decimal_roundoff)
    #ensure that the instances are are still [0,2pi)
    instance_array = instance_array % (2.*np.pi)
    print np.max(instance_array)>(2.*np.pi), np.min(instance_array)<0
    #prepare seeds if they weren't provided
    if seed_list==None:
        seed_list = map(int, np.round(np.random.rand(number_of_starts)*100.))
    multiple_run_results = {}; q_val_list = []
    if n_cpus>1:
        pool_size = n_cpus
        print '  pool size :', pool_size
        pool = multiprocessing.Pool(processes=pool_size)
        results = pool.map(_k_means_p_multiproc_wrapper, 
                           itertools.izip(itertools.repeat(k), seed_list, itertools.repeat(instance_array),
                                          itertools.repeat(distance_calc), itertools.repeat(convergence_diff_cutoff),
                                          itertools.repeat(n_iterations)))
        print '  closing pool and waiting for pool to finish'
        pool.close(); pool.join() # no more tasks
        print '  pool finished'
    else:
        results = []
        for seed in seed_list:
            multiple_run_results[seed] = {}
            results.append(_k_means_p_single_seed(k, seed, instance_array, distance_calc, convergence_diff_cutoff, n_iterations))
    #put all the results in a dictionary... necessary step???
    for i in range(0,len(results)):
        multiple_run_results[seed_list[i]] = {}
        multiple_run_results[seed_list[i]]['cluster_assignments'] = results[i][1]
        multiple_run_results[seed_list[i]]['centroids'] = results[i][0]
        multiple_run_results[seed_list[i]]['q_val'] = results[i][2]
        q_val_list.append(results[i][2])
    #pick out the best answer from the runs
    print q_val_list, np.argmin(q_val_list)
    seed_best = seed_list[np.argmin(q_val_list)]
    print 'Best seed {seed}'.format(seed=seed_best)
    cluster_details = {'k_means_periodic_means':multiple_run_results[seed_best]['centroids'], 'k_means_periodic_q_val':multiple_run_results[seed_best]['q_val']}
    return multiple_run_results[seed_best]['cluster_assignments'], cluster_details

###############################################################
#############################EM-VM##############################
def _EM_VMM_check_convergence(mu_list_old, mu_list_new, kappa_list_old, kappa_list_new):
    return np.sqrt(np.sum((mu_list_old - mu_list_new)**2)), np.sqrt(np.sum((kappa_list_old - kappa_list_new)**2))

def _EM_VMM_maximise_single_cluster(input_arguments):
    cluster_ident, instance_array, assignments = input_arguments
    current_datapoints = (assignments==cluster_ident)
    print os.getpid(), 'Maximisation step, cluster:%d, dimension:'%(cluster_ident,),
    mu_list_cluster = []
    kappa_list_cluster = []
    n_dimensions = instance_array.shape[1]
    if np.sum(current_datapoints)>10:
        for dim_loc in range(n_dimensions):
            #print '%d'%(dim_loc),
            #kappa_tmp, loc_tmp, scale_fit = vonmises.fit(instance_array[current_datapoints, dim_loc],fscale=1)
            kappa_tmp, loc_tmp, scale_fit = EM_VMM_calc_best_fit(instance_array[current_datapoints, dim_loc])
            #update to the best fit parameters
            mu_list_cluster.append(loc_tmp)
            kappa_list_cluster.append(kappa_tmp)
        success = 1
    else:
        success = 0;mu_list_cluster = []; kappa_list_cluster = []
    print ''
    return np.array(mu_list_cluster), np.array(kappa_list_cluster),cluster_ident,success

def kappa_guess_func(kappa,R_e):
    return (R_e - spec.iv(1,kappa)/spec.iv(0,kappa))**2

def EM_VMM_calc_best_fit_optimise(z,lookup=None,N=None):
    '''Calculates MLE approximate parameters for mean and kappa for
    the von Mises distribution. Can use a lookup table for the two Bessel
    functions, or a scipy optimiser if lookup=None

    SH: 23May2013 '''
    if N==None:
        N = len(z)
    z_bar = np.sum(z)/float(N)
    mean_theta = np.angle(z_bar)
    R_squared = np.real(z_bar * z_bar.conj())
    tmp = (float(N)/(float(N)-1))*(R_squared-1./float(N))
    #This is to catch problems with the sqrt below - however, need to track down why this happens...
    #This happens for very low kappa values - i.e terrible clusters...
    if tmp<0:tmp = 0.
    R_e = np.sqrt(tmp)
    if lookup==None:
        tmp1 = opt.fmin(kappa_guess_func,3,args=(R_e,),disp=0)
        kappa = tmp1[0]
    else:
        min_arg = np.argmin(np.abs(lookup[0]-R_e))
        kappa = lookup[1][min_arg]
    return kappa, mean_theta, 1

def EM_VMM_calc_best_fit(z,N=None,lookup=None):
    '''Calculates MLE approximate parameters for mean and kappa for
    the von Mises distribution. Can use a lookup table for the two Bessel
    functions, or a scipy optimiser if lookup=None

    SH: 23May2013 '''
    if N==None:
        N = len(z)
    z_bar = np.sum(z,axis=0)/float(N)
    mean_theta = np.angle(z_bar)
    R_bar = np.abs(z_bar)
    if len(R_bar.shape)==0:
        if R_bar<0.53:
            kappa = 2.* R_bar + R_bar**3 + 5./6*R_bar**5
            #print 'approx 1'
        elif R_bar<0.85:
            kappa = -0.4 + 1.39*R_bar + 0.43/(1-R_bar)
            #print 'approx 2'
        elif R_bar<=1:
            kappa = 1./(2.*(1-R_bar))
            #print 'approx 3'
        else:
            raise ValueError()
    else:
        #kappa = R_bar*0.
        kappa = 1./(2.*(1-R_bar))
        #kappa[R_bar<=1.0] = 1./(2.*(1-R_bar[R_bar<=1.0]))
        kappa[R_bar<0.85] = -0.4 + 1.39*R_bar[R_bar<0.85] + 0.43/(1-R_bar[R_bar<0.85])
        kappa[R_bar<0.53] = 2.* R_bar[R_bar<0.53] + R_bar[R_bar<0.53]**3 + (5./6)*(R_bar[R_bar<0.53])**5
        #kappa= np.array(kappa)
    return kappa, mean_theta, 1



def EM_GMM_calc_best_fit(instance_array,weights):
    '''Calculates MLE approximate parameters for mean and stddev for
    the Gaussian distribution. 

    SH: 23May2013 '''
    N = np.sum(weights)
    #z = (instance_array.T * weights).T
    z = (instance_array * weights[:,np.newaxis])
    #print 'hello', np.allclose(z, z2)
    mean_theta = np.sum(z,axis=0)/float(N)
    sigma = np.sqrt(1./N *np.sum(weights[:,np.newaxis] * (instance_array - mean_theta)**2, axis = 0))
    return sigma, mean_theta

def EM_gamma_calc_best_fit(instance_array, weights):
    '''Calculates MLE approximate parameters for mean and stddev for
    the Gaussian distribution. 

    SH: 23May2013 '''
    N = np.sum(weights)
    #z = (instance_array.T * weights).T
    z = (instance_array * weights[:,np.newaxis])
    #print 'hello', np.allclose(z, z2)
    s = np.log(np.sum(z,axis=0)/float(N)) - np.sum(np.log(z),axis=0)/float(N)
    k = (3. - s + np.sqrt((s-3)**2 + 24*s))/(12.*s)
    theta = np.sum(z,axis=0)/float(N)/k
    return k, theta



#We can do this step in parallel....
#Either parallel over clusters, or parallel over dimensions....
def _EM_VMM_maximisation_step_hard(mu_list, kappa_list, instance_array, assignments, instance_array_complex = None, bessel_lookup_table = None, n_cpus=1):
    if instance_array_complex==None:
        instance_array_complex = np.exp(1j*instance_array)
    n_clusters = len(mu_list)
    n_datapoints, n_dimensions = instance_array.shape
    mu_list_old = copy.deepcopy(mu_list)
    kappa_list_old = copy.deepcopy(kappa_list)
    start_time = time.time()
    if n_cpus>1:
        print 'creating pool map ', n_cpus
        pool = multiprocessing.Pool(processes = n_cpus, maxtasksperchild=2)
        #output_data = pool.map(_EM_VMM_maximise_single_cluster, itertools.izip(range(n_clusters), itertools.repeat(instance_array),itertools.repeat(assignments)))
        output_data = pool.map(_EM_VMM_maximise_single_cluster, itertools.izip(range(n_clusters), itertools.repeat(instance_array),itertools.repeat(assignments)))
        pool.close(); pool.join()
        for mu_list_cluster, kappa_list_cluster, cluster_ident, success in output_data:
            if success==1:
                mu_list[cluster_ident][:]=mu_list_cluster
                kappa_list[cluster_ident][:]=kappa_list_cluster
                
    else:
        for cluster_ident in range(n_clusters):
            current_datapoints = (assignments==cluster_ident)
            #Only try to fit the von Mises distribution if there
            #are datapoints!!!!
            #print 'Maximisation step, cluster:%d, dimension:'%(cluster_ident,),
            if np.sum(current_datapoints)>0:
                for dim_loc in range(n_dimensions):
                    #print '%d-'%(dim_loc),
                    #kappa_tmp, loc_tmp, scale_fit = vonmises.fit(instance_array[current_datapoints, dim_loc],fscale=1)
                    #kappa_tmp, loc_tmp, scale_fit = calc_best_fit(instance_array[current_datapoints, dim_loc])
                    kappa_tmp, loc_tmp, scale_fit = EM_VMM_calc_best_fit(instance_array_complex[current_datapoints, dim_loc],lookup=bessel_lookup_table)
                    #update to the best fit parameters
                    mu_list[cluster_ident][dim_loc]=loc_tmp
                    kappa_list[cluster_ident][dim_loc]=kappa_tmp
    convergence_mu, convergence_kappa = _EM_VMM_check_convergence(mu_list_old, mu_list, kappa_list_old, kappa_list)
    print 'maximisation time: %.2f'%(time.time()-start_time)
    return mu_list, kappa_list, convergence_mu, convergence_kappa


def _EM_VMM_maximisation_step_soft(mu_list, kappa_list, instance_array, zij, instance_array_complex = None, bessel_lookup_table = None, n_cpus=1):
    n_clusters = len(mu_list)
    n_datapoints, n_dimensions = instance_array.shape
    pi_hat = np.sum(zij,axis=0)/float(n_datapoints)
    if instance_array_complex==None:
        instance_array_complex = np.exp(1j*instance_array)
    mu_list_old = copy.deepcopy(mu_list)
    kappa_list_old = copy.deepcopy(kappa_list)

    for cluster_ident in range(n_clusters):
        inst_tmp = (instance_array_complex.T * zij[:,cluster_ident]).T
        N= np.sum(zij[:,cluster_ident])

        #calculate the best fit for this cluster - all dimensions at once.... using new approximations
        kappa_tmp1, loc_tmp1, scale_fit1 = EM_VMM_calc_best_fit(inst_tmp, lookup=bessel_lookup_table,N=N)
        mu_list[cluster_ident] = loc_tmp1
        kappa_list[cluster_ident] = kappa_tmp1
        # for dim_loc in range(n_dimensions):
        #     #Only do this if there are items to cluster!!
        #     if N>=5:
        #         kappa_tmp, loc_tmp, scale_fit = EM_VMM_calc_best_fit(inst_tmp[:, dim_loc], lookup=bessel_lookup_table,N=N)
        #         #update to the best fit parameters
        #         mu_list[cluster_ident][dim_loc]=loc_tmp
        #         kappa_list[cluster_ident][dim_loc]=kappa_tmp
        #         #print '{:.2e},{:.2e}'.format(np.max(np.abs(loc_tmp - loc_tmp1[dim_loc])), np.max(np.abs(kappa_tmp - kappa_tmp1[dim_loc]))),
        # #print ''
        # print np.max(np.abs(np.array(mu_list[cluster_ident]) - loc_tmp1)), np.max(np.abs(np.array(kappa_list[cluster_ident]) - kappa_tmp1))

    kappa_list = np.clip(kappa_list,0.1,300)
    convergence_mu, convergence_kappa = _EM_VMM_check_convergence(mu_list_old, mu_list, kappa_list_old, kappa_list)
    #print 'maximisation times: %.2f'%(time.time()-start_time)
    return mu_list, kappa_list, pi_hat, convergence_mu, convergence_kappa

def _EM_VMM_expectation_step_hard(mu_list, kappa_list, instance_array):
    start_time = time.time()
    n_clusters = len(mu_list); 
    n_datapoints, n_dimensions = instance_array.shape
    probs = np.ones((instance_array.shape[0],n_clusters),dtype=float)
    for mu_tmp, kappa_tmp, cluster_ident in zip(mu_list,kappa_list,range(n_clusters)):
        #We are checking the probability of belonging to cluster_ident
        probs_1 = np.product(np.exp(kappa_tmp*np.cos(instance_array-mu_tmp))/(2.*np.pi*spec.iv(0,kappa_tmp)),axis=1)
        probs[:,cluster_ident] = probs_1
    assignments = np.argmax(probs,axis=1)
    #return assignments, L
    return assignments, 0

def _EM_VMM_expectation_step_soft(mu_list, kappa_list, instance_array, pi_hat, c_arr = None, s_arr = None):
    n_clusters = len(mu_list); 
    n_datapoints, n_dimensions = instance_array.shape
    probs = np.ones((instance_array.shape[0],n_clusters),dtype=float)

    #c_arr and s_arr are used to speed up cos(instance_array - mu) using
    #the trig identity cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
    #this removes the need to constantly recalculate cos(a) and sin(a)
    if c_arr==None or s_arr==None:
        c_arr = np.cos(instance_array)
        s_arr = np.sin(instance_array)

    # c_arr2 = c_arr[:,np.newaxis,:]
    # s_arr2 = s_arr[:,np.newaxis,:]
    # pt1 = (np.cos(mu_list))[np.newaxis,:,:]
    # pt2  = (np.sin(mu_list))[np.newaxis,:,:]
    # pt2 = (c_arr2*pt1 + s_arr2*pt2)
    # pt2 = (np.array(kappa_list))[np.newaxis,:,:] * pt2
    #print pt2.shape
    for mu_tmp, kappa_tmp, p_hat, cluster_ident in zip(mu_list,kappa_list,pi_hat,range(n_clusters)):
        norm_fac_exp = len(mu_list[0])*np.log(1./(2.*np.pi)) - np.sum(np.log(spec.iv(0,kappa_tmp)))
        pt1 = kappa_tmp * (c_arr*np.cos(mu_tmp) + s_arr*np.sin(mu_tmp))
        probs[:,cluster_ident] = p_hat * np.exp(np.sum(pt1,axis=1) + norm_fac_exp)

        #old way without trig identity speed up
        #probs[:,cluster_ident] = p_hat * np.exp(np.sum(kappa_tmp*np.cos(instance_array - mu_tmp),axis=1)+norm_fac_exp)

        #older way including everything in exponent, and not taking log of hte constant
        #probs[:,cluster_ident] = p_hat * np.product( np.exp(kappa_tmp*np.cos(instance_array-mu_tmp))/(2.*np.pi*spec.iv(0,kappa_tmp)),axis=1)
    prob_sum = (np.sum(probs,axis=1))[:,np.newaxis]
    zij = probs/((prob_sum))

    #This was from before using np.newaxis
    #zij = (probs.T/prob_sum).T

    #Calculate the log-likelihood - note this is quite an expensive computation and not really necessary
    #unless comparing different techniques and/or checking for convergence
    valid_items = probs>1.e-20
    #L = np.sum(zij[probs>1.e-20]*np.log(probs[probs>1.e-20]))
    L = np.sum(zij[valid_items]*np.log(probs[valid_items]))
    #L = np.sum(zij*np.log(np.clip(probs,1.e-10,1)))
    return zij, L


def EM_VMM_clustering(instance_array, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random'):
    #This is for the new method...
    instance_array_complex = np.exp(1j*instance_array)

    kappa_lookup = np.linspace(0,100,10000)
    bessel_lookup_table = [spec.iv(1,kappa_lookup)/spec.iv(0,kappa_lookup), kappa_lookup]
    print '...'

    n_dimensions = instance_array.shape[1]
    iteration = 1    
    #First assignment step
    mu_list = np.ones((n_clusters,n_dimensions),dtype=float)
    kappa_list = np.ones((n_clusters,n_dimensions),dtype=float)
    LL_list = []
    if start=='k_means':
        print 'Initialising clusters using a fast k_means run'
        cluster_assignments, cluster_details = k_means_clustering(instance_array, n_clusters=n_clusters, sin_cos = 1, number_of_starts = 1)
        mu_list, kappa_list, convergence_mu, convergence_kappa = _EM_VMM_maximisation_step_hard(mu_list, kappa_list, instance_array, cluster_assignments, instance_array_complex = instance_array_complex, bessel_lookup_table= bessel_lookup_table, n_cpus=n_cpus)
    elif start=='EM_GMM':
        print 'Initialising clusters using a EM_GMM run'
        cluster_assignments, cluster_details = EM_GMM_clustering(instance_array, n_clusters=n_clusters, sin_cos = 0, number_of_starts = 1)
        mu_list, kappa_list, convergence_mu, convergence_kappa = _EM_VMM_maximisation_step_hard(mu_list, kappa_list, instance_array, cluster_assignments, instance_array_complex = instance_array_complex, bessel_lookup_table = bessel_lookup_table, n_cpus=n_cpus)
    else:
        print 'Initialising clusters using random start points'
        mu_list = np.random.rand(n_clusters,n_dimensions)*2.*np.pi - np.pi
        kappa_list = np.random.rand(n_clusters,n_dimensions)*20
        cluster_assignments, L = _EM_VMM_expectation_step_hard(mu_list, kappa_list,instance_array)
        while np.min([np.sum(cluster_assignments==i) for i in range(len(mu_list))])<20:#(instance_array.shape[0]/n_clusters/4):
            print 'recalculating initial points'
            mu_list = np.random.rand(n_clusters,n_dimensions)*2.*np.pi - np.pi
            kappa_list = np.random.rand(n_clusters,n_dimensions)*20
            cluster_assignments = _EM_VMM_expectation_step_hard(mu_list, kappa_list,instance_array)
            print cluster_assignments
    convergence_record = []
    converged = 0; 
    while (iteration<=n_iterations) and converged!=1:
        start_time = time.time()
        cluster_assignments, L = _EM_VMM_expectation_step_hard(mu_list, kappa_list,instance_array)
        LL_list.append(L)
        mu_list, kappa_list, convergence_mu, convergence_kappa = _EM_VMM_maximisation_step_hard(mu_list, kappa_list, instance_array, cluster_assignments,  instance_array_complex = instance_array_complex, bessel_lookup_table = bessel_lookup_table, n_cpus=n_cpus)
        print 'Time for iteration %d :%.2f, mu_convergence:%.3f, kappa_convergence:%.3f, LL: %.8e'%(iteration,time.time() - start_time,convergence_mu, convergence_kappa,L)
        convergence_record.append([iteration, convergence_mu, convergence_kappa])
        if convergence_mu<0.01 and convergence_kappa<0.01:
            converged = 1
            print 'Convergence criteria met!!'
        iteration+=1
    print 'AIC : %.2f'%(2*(mu_list.shape[0]*mu_list.shape[1])-2.*LL_list[-1])
    cluster_details = {'EM_VMM_means':mu_list, 'EM_VMM_kappas':kappa_list, 'EM_VMM_LL':LL_list}
    return cluster_assignments, cluster_details


def EM_VMM_clustering_soft(instance_array, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random',bessel_lookup_table=True):
    '''
    Expectation maximisation using von Mises with soft cluster
    assignments.  instance_array : the input phases n_clusters :
    number of clusters to aim for n_iterations : number of iterations
    before giving up n_cpus : currently not implemented start: how to
    start the clusters off - recommend using 'k_means'
    bessel_lookup_table : how to calculate kappa, can use a lookup
    table or optimiser

    SH : 23May2013
    '''

    instance_array_complex = np.exp(1j*instance_array)
    instance_array_c = np.real(instance_array_complex)
    instance_array_s = np.imag(instance_array_complex)
    n_dimensions = instance_array.shape[1]
    iteration = 1    
    if bessel_lookup_table:
        kappa_lookup = np.linspace(0,100,10000)
        bessel_lookup_table = [spec.iv(1,kappa_lookup)/spec.iv(0,kappa_lookup), kappa_lookup]
    else:
        bessel_lookup_table=None
    #First assignment step
    mu_list = np.ones((n_clusters,n_dimensions),dtype=float)
    kappa_list = np.ones((n_clusters,n_dimensions),dtype=float)
    LL_list = []
    zij = np.zeros((instance_array.shape[0],n_clusters),dtype=float)
    if start=='k_means':
        print 'Initialising clusters using a fast k_means run'
        cluster_assignments, cluster_details = k_means_clustering(instance_array, n_clusters=n_clusters, sin_cos = 1, number_of_starts = 3)
        for i in list(set(cluster_assignments)):
            zij[cluster_assignments==i,i] = 1
        #print zij
        print 'finished initialising'
    elif start=='EM_GMM':
        cluster_assignments, cluster_details = EM_GMM_clustering(instance_array, n_clusters=n_clusters, sin_cos = 1, number_of_starts = 1)
        for i in list(set(cluster_assignments)):
            zij[cluster_assignments==i,i] = 1
    else:
        print 'going with random option'
        zij = np.random.random(zij.shape)
    mu_list, kappa_list, pi_hat, convergence_mu, convergence_kappa = _EM_VMM_maximisation_step_soft(mu_list, kappa_list, instance_array, zij, instance_array_complex = instance_array_complex, bessel_lookup_table= bessel_lookup_table, n_cpus=n_cpus)
        
    convergence_record = []
    converged = 0; 
    LL_diff = np.inf
    while (iteration<=n_iterations) and converged!=1:
        start_time = time.time()
        zij, L = _EM_VMM_expectation_step_soft(mu_list, kappa_list, instance_array, pi_hat, c_arr = instance_array_c, s_arr = instance_array_s)
        LL_list.append(L)
        cluster_assignments = np.argmax(zij,axis=1)
        mu_list, kappa_list, pi_hat, convergence_mu, convergence_kappa = _EM_VMM_maximisation_step_soft(mu_list, kappa_list, instance_array, zij, instance_array_complex = instance_array_complex, bessel_lookup_table= bessel_lookup_table, n_cpus=n_cpus)
        if (iteration>=2): LL_diff = np.abs(((LL_list[-1] - LL_list[-2])/LL_list[-2]))
        #print 'Time for iteration %d :%.2f, mu_convergence:%.3e, kappa_convergence:%.3e, LL: %.8e, LL_dif : %.3e'%(iteration,time.time() - start_time,convergence_mu, convergence_kappa,L,LL_diff)
        convergence_record.append([iteration, convergence_mu, convergence_kappa])
        if iteration>200 and LL_diff <0.0001:
           converged = 1
           print 'Convergence criteria met!!'
        iteration+=1
    print 'Time for iteration %d :%.2f, mu_convergence:%.3e, kappa_convergence:%.3e, LL: %.8e, LL_dif : %.3e'%(iteration,time.time() - start_time,convergence_mu, convergence_kappa,L,LL_diff)
    #print 'AIC : %.2f'%(2*(mu_list.shape[0]*mu_list.shape[1])-2.*LL_list[-1])
    cluster_assignments = np.argmax(zij,axis=1)
    cluster_details = {'EM_VMM_means':mu_list, 'EM_VMM_kappas':kappa_list, 'EM_VMM_LL':LL_list, 'zij':zij}
    return cluster_assignments, cluster_details

def EM_VMM_clustering_wrapper2(input_data):
    tmp = EM_VMM_clustering_class(*input_data)
    return copy.deepcopy(tmp.cluster_assignments), copy.deepcopy(tmp.cluster_details)


def EM_VMM_clustering_wrapper(instance_array, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random', kappa_calc='approx', hard_assignments = 0, kappa_converged = 0.1, mu_converged = 0.01, min_iterations=10, LL_converged = 1.e-4, verbose = 0, number_of_starts = 1, seeds = None):
    cluster_list = [n_clusters for i in range(number_of_starts)]
    if seeds == None:
        seed_list = [int(np.random.rand()*10000) for i in range(number_of_starts)]# )[None]*number_of_starts
        print 'Seeds: ', seed_list
    elif (seeds.__class__ == int) and (number_of_starts==1):
        seed_list = [seed_list]
    elif (seeds.__class__ == int) and (number_of_starts!=1):
        raise Exception('Only one seed given for more than one start - This will give duplicate results, fix and try again!!!')
    ## Added by JG
    else:
        seed_list = seeds
    ##
    if len(seed_list)!=number_of_starts:
        raise Exception('The length of the seed list is different to the number of starts - fix and try again!!!')

    #seed_list = [i for i in range(number_of_starts)]
    rep = itertools.repeat
    from multiprocessing import Pool
    input_data_iter = itertools.izip(rep(instance_array), rep(n_clusters),
                                     rep(n_iterations), rep(n_cpus), rep(start), rep(kappa_calc),
                                     rep(hard_assignments), rep(kappa_converged),
                                     rep(mu_converged),rep(min_iterations), rep(LL_converged),
                                     rep(verbose), seed_list)
    if n_cpus>1:
        pool_size = n_cpus
        pool = Pool(processes=pool_size, maxtasksperchild=3)
        print 'creating pool map'
        results = pool.map(EM_VMM_clustering_wrapper2, input_data_iter)
        print 'waiting for pool to close '
        pool.close()
        print 'joining pool'
        pool.join()
        print 'pool finished'
    else:
        results = map(EM_VMM_clustering_wrapper2, input_data_iter)
    LL_results = []
    for tmp in results: LL_results.append(tmp[1]['LL'][-1])
    print("I'm in here JG", LL_results)
    tmp_loc = np.argmax(LL_results)
    return results[tmp_loc]

class EM_VMM_clustering_class():
    def __init__(self, instance_array, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random', kappa_calc='approx', hard_assignments = 0, kappa_converged = 0.1, mu_converged = 0.01, min_iterations=10, LL_converged = 1.e-4, verbose = 0, seed=None):
        '''
        Expectation maximisation using von Mises with soft cluster
        assignments.  instance_array : the input phases n_clusters :
        number of clusters to aim for n_iterations : number of iterations
        before giving up n_cpus : currently not implemented start: how to
        start the clusters off - recommend using 'k_means'
        bessel_lookup_table : how to calculate kappa, can use a lookup
        table or optimiser

        kappa_calc : approx, lookup_table, optimize
        SH : 23May2013
        '''
        #min iterations, max iterations
        #kappa change, mu change
        self.instance_array = copy.deepcopy(instance_array)
        self.instance_array_complex = np.exp(1j*self.instance_array)
        self.instance_array_c = np.real(self.instance_array_complex)
        self.instance_array_s = np.imag(self.instance_array_complex)
        self.n_instances, self.n_dimensions = self.instance_array.shape
        self.n_clusters = n_clusters
        self.max_iterations = n_iterations
        self.start = start
        self.hard_assignments = hard_assignments
        self.seed = seed
        #If None is given as a seed, get a random number between 1 and 10000
        if self.seed == None:
            self.seed=int(round(np.random.rand()*10000))+1
            #self.seed = os.getpid()
        print('seed,',self.seed)
        np.random.seed(self.seed)
        if kappa_calc == 'lookup_table':
            self.generate_bessel_lookup_table()
        else:
            self.bessel_lookup_table=None
        self.iteration = 1
        self.initialisation()
        self.convergence_record = []
        converged = 0; 
        self.LL_diff = np.inf
        while converged!=1:
            start_time = time.time()
            self._EM_VMM_expectation_step()
            if self.hard_assignments:
                print 'hard assignments'
                self.cluster_assignments = np.argmax(self.zij,axis=1)
                self.zij = self.zij *0
                for i in range(self.n_clusters):
                    self.zij[self.cluster_assignments==i,i] = 1

            valid_items = self.probs>(1.e-300)
            self.LL_list.append(np.sum(self.zij[valid_items]*np.log(self.probs[valid_items])))
            self._EM_VMM_maximisation_step()
            if (self.iteration>=2): self.LL_diff = np.abs(((self.LL_list[-1] - self.LL_list[-2])/self.LL_list[-2]))
            if verbose:
                print 'Time for iteration %d :%.2f, mu_convergence:%.3e, kappa_convergence:%.3e, LL: %.8e, LL_dif : %.3e'%(self.iteration,time.time() - start_time,self.convergence_mu, self.convergence_kappa,self.LL_list[-1],self.LL_diff)
            self.convergence_record.append([self.iteration, self.convergence_mu, self.convergence_kappa])
            if (self.iteration > min_iterations) and (self.convergence_mu<mu_converged) and (self.convergence_kappa<kappa_converged) and (self.LL_diff<LL_converged):
                converged = 1
                print 'Convergence criteria met!!'
            elif self.iteration > n_iterations:
                converged = 1
                print 'Max number of iterations'
            self.iteration+=1
        print os.getpid(), 'Time for iteration %d :%.2f, mu_convergence:%.3e, kappa_convergence:%.3e, LL: %.8e, LL_dif : %.3e'%(self.iteration,time.time() - start_time,self.convergence_mu, self.convergence_kappa,self.LL_list[-1],self.LL_diff)
        #print 'AIC : %.2f'%(2*(mu_list.shape[0]*mu_list.shape[1])-2.*LL_list[-1])
        self.cluster_assignments = np.argmax(self.zij,axis=1)
        self.BIC = -2*self.LL_list[-1]+self.n_clusters*3*np.log(self.n_dimensions)
        #self.cluster_details = {'EM_VMM_means':self.mu_list, 'EM_VMM_kappas':self.kappa_list, 'LL':self.LL_list, 'zij':self.zij, 'BIC':self.BIC}
        self.cluster_details = {'EM_VMM_means':self.mu_list, 'EM_VMM_kappas':self.kappa_list, 'LL':self.LL_list, 'BIC':self.BIC}

    def generate_bessel_lookup_table(self):
        self.kappa_lookup = np.linspace(0,100,10000)
        self.bessel_lookup_table = [spec.iv(1,kappa_lookup)/spec.iv(0,kappa_lookup), kappa_lookup]

    def initialisation(self):
        '''This involves generating the mu and kappa arrays
        Then initialising based on self.start using k-means, EM-GMM or
        giving every instance a random probability of belonging to each cluster
        SH: 7June2013
        '''
        self.mu_list = np.ones((self.n_clusters,self.n_dimensions),dtype=float)
        self.kappa_list = np.ones((self.n_clusters,self.n_dimensions),dtype=float)
        self.LL_list = []
        self.zij = np.zeros((self.instance_array.shape[0],self.n_clusters),dtype=float)
        if self.start=='k_means':
            print 'Initialising clusters using a fast k_means run'
            self.cluster_assignments, self.cluster_details = k_means_clustering(self.instance_array, n_clusters=self.n_clusters, sin_cos = 1, number_of_starts = 3, seed=self.seed)
            for i in list(set(self.cluster_assignments)):
                self.zij[self.cluster_assignments==i,i] = 1
            print 'finished initialising'
        elif self.start=='EM_GMM':
            self.cluster_assignments, self.cluster_details = EM_GMM_clustering(self.instance_array, n_clusters=self.n_clusters, sin_cos = 1, number_of_starts = 1)
            for i in list(set(cluster_assignments)):
                self.zij[cluster_assignments==i,i] = 1
        else:
            print 'going with random option'
            #need to get this to work better.....
            self.zij = np.random.random(self.zij.shape)
            #and normalise so each row adds up to 1....
            self.zij = self.zij / ((np.sum(self.zij,axis=1))[:,np.newaxis])
        self._EM_VMM_maximisation_step()

    def _EM_VMM_maximisation_step(self):
        '''Maximisation step SH : 7June2013
        '''
        self.pi_hat = np.sum(self.zij,axis=0)/float(self.n_instances)
        self.mu_list_old = self.mu_list.copy()
        self.kappa_list_old = self.kappa_list.copy()
        for cluster_ident in range(self.n_clusters):
            inst_tmp = (self.instance_array_complex.T * self.zij[:,cluster_ident]).T
            N= np.sum(self.zij[:,cluster_ident])
            #calculate the best fit for this cluster - all dimensions at once.... using new approximations
            self.kappa_list[cluster_ident,:], self.mu_list[cluster_ident,:], scale_fit1 = EM_VMM_calc_best_fit(inst_tmp, lookup=self.bessel_lookup_table, N=N)
        #Prevent ridiculous situations happening....
        self.kappa_list = np.clip(self.kappa_list,0.1,300)
        self._EM_VMM_check_convergence()

    def _EM_VMM_check_convergence(self,):
        self.convergence_mu = np.sqrt(np.sum((self.mu_list_old - self.mu_list)**2))
        self.convergence_kappa = np.sqrt(np.sum((self.kappa_list_old - self.kappa_list)**2))

    def _EM_VMM_expectation_step(self,):
        self.probs = self.zij*0#np.ones((self.instance_array.shape[0],self.n_clusters),dtype=float)
        #instance_array_c and instance_array_s are used to speed up cos(instance_array - mu) using
        #the trig identity cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
        #this removes the need to constantly recalculate cos(a) and sin(a)
        for mu_tmp, kappa_tmp, p_hat, cluster_ident in zip(self.mu_list,self.kappa_list,self.pi_hat,range(self.n_clusters)):
            #norm_fac_exp = self.n_clusters*np.log(1./(2.*np.pi)) - np.sum(np.log(spec.iv(0,kappa_tmp)))
            norm_fac_exp = -self.n_dimensions*np.log(2.*np.pi) - np.sum(np.log(spec.iv(0,kappa_tmp)))
            pt1 = kappa_tmp * (self.instance_array_c*np.cos(mu_tmp) + self.instance_array_s*np.sin(mu_tmp))
            self.probs[:,cluster_ident] = p_hat * np.exp(np.sum(pt1,axis=1) + norm_fac_exp)
        prob_sum = (np.sum(self.probs,axis=1))[:,np.newaxis]
        self.zij = self.probs/(prob_sum)
        #Calculate the log-likelihood - note this is quite an expensive computation and not really necessary
        #unless comparing different techniques and/or checking for convergence
        #This is to prevent problems with log of a very small number....
        #L = np.sum(zij[probs>1.e-20]*np.log(probs[probs>1.e-20]))
        #L = np.sum(zij*np.log(np.clip(probs,1.e-10,1)))







def EM_GMM_clustering_wrapper2(input_data):
    tmp = EM_GMM_clustering_class(*input_data)
    return copy.deepcopy(tmp.cluster_assignments), copy.deepcopy(tmp.cluster_details)


def EM_GMM_clustering_wrapper(instance_array, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random', kappa_calc='approx', hard_assignments = 0, kappa_converged = 0.1, mu_converged = 0.01, min_iterations=10, LL_converged = 1.e-4, verbose = 0, number_of_starts = 1):
    cluster_list = [n_clusters for i in range(number_of_starts)]
    seed_list = [i for i in range(number_of_starts)]
    rep = itertools.repeat
    from multiprocessing import Pool
    input_data_iter = itertools.izip(rep(instance_array), rep(n_clusters),
                                     rep(n_iterations), rep(n_cpus), rep(start), rep(kappa_calc),
                                     rep(hard_assignments), rep(kappa_converged),
                                     rep(mu_converged),rep(min_iterations), rep(LL_converged),
                                     rep(verbose), seed_list)
    if n_cpus>1:
        pool_size = n_cpus
        pool = Pool(processes=pool_size, maxtasksperchild=3)
        print 'creating pool map'
        results = pool.map(EM_GMM_clustering_wrapper2, input_data_iter)
        print 'waiting for pool to close '
        pool.close()
        print 'joining pool'
        pool.join()
        print 'pool finished'
    else:
        results = map(EM_GMM_clustering_wrapper2, input_data_iter)
    LL_results = []
    for tmp in results: LL_results.append(tmp[1]['LL'][-1])
    print LL_results
    tmp_loc = np.argmax(LL_results)
    return results[tmp_loc]

class EM_GMM_clustering_class():
    def __init__(self, instance_array, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random', kappa_calc='approx',hard_assignments = 0, kappa_converged = 0.1, mu_converged = 0.01, min_iterations=10, LL_converged = 1.e-4, verbose = 0, seed=None):
        '''
        Expectation maximisation using von Mises with soft cluster
        assignments.  instance_array : the input phases n_clusters :
        number of clusters to aim for n_iterations : number of iterations
        before giving up n_cpus : currently not implemented start: how to
        start the clusters off - recommend using 'k_means'
        bessel_lookup_table : how to calculate kappa, can use a lookup
        table or optimiser

        kappa_calc : approx, lookup_table, optimize
        SH : 23May2013
        '''
        #min iterations, max iterations
        #kappa change, mu change
        self.instance_array = copy.deepcopy(instance_array)
        #self.instance_array_complex = np.exp(1j*self.instance_array)
        #self.instance_array_c = np.real(self.instance_array_complex)
        #self.instance_array_s = np.imag(self.instance_array_complex)
        self.n_instances, self.n_dimensions = self.instance_array.shape
        self.n_clusters = n_clusters
        self.max_iterations = n_iterations
        self.start = start
        self.hard_assignments = hard_assignments
        self.seed = seed
        if self.seed == None:
            self.seed = os.getpid()
        print('seed,',self.seed)
        np.random.seed(self.seed)
        self.iteration = 1
        self.initialisation()
        self.convergence_record = []
        converged = 0; 
        self.LL_diff = np.inf
        while converged!=1:
            start_time = time.time()
            self._EM_GMM_expectation_step()
            if self.hard_assignments:
                print 'hard assignments'
                self.cluster_assignments = np.argmax(self.zij,axis=1)
                self.zij = self.zij *0
                for i in range(self.n_clusters):
                    self.zij[self.cluster_assignments==i,i] = 1

            valid_items = self.probs>(1.e-300)
            self.LL_list.append(np.sum(self.zij[valid_items]*np.log(self.probs[valid_items])))
            self._EM_GMM_maximisation_step()
            if (self.iteration>=2): self.LL_diff = np.abs(((self.LL_list[-1] - self.LL_list[-2])/self.LL_list[-2]))
            if verbose:
                print 'Time for iteration %d :%.2f, mu_convergence:%.3e, kappa_convergence:%.3e, LL: %.8e, LL_dif : %.3e'%(self.iteration,time.time() - start_time,self.convergence_mu, self.convergence_kappa,self.LL_list[-1],self.LL_diff)
            self.convergence_record.append([self.iteration, self.convergence_mu, self.convergence_kappa])
            if (self.iteration > min_iterations) and (self.convergence_mu<mu_converged) and (self.convergence_kappa<kappa_converged) and (self.LL_diff<LL_converged):
                converged = 1
                print 'Convergence criteria met!!'
            elif self.iteration > n_iterations:
                converged = 1
                print 'Max number of iterations'
            self.iteration+=1
        print os.getpid(), 'Time for iteration %d :%.2f, mu_convergence:%.3e, kappa_convergence:%.3e, LL: %.8e, LL_dif : %.3e'%(self.iteration,time.time() - start_time,self.convergence_mu, self.convergence_kappa,self.LL_list[-1],self.LL_diff)
        #print 'AIC : %.2f'%(2*(mu_list.shape[0]*mu_list.shape[1])-2.*LL_list[-1])
        self.cluster_assignments = np.argmax(self.zij,axis=1)
        self.BIC = -2*self.LL_list[-1]+self.n_clusters*3*np.log(self.n_dimensions)
        self.cluster_details = {'EM_VMM_means':self.mu_list, 'EM_VMM_kappas':self.std_list, 'LL':self.LL_list, 'zij':self.zij, 'BIC':self.BIC}

    def initialisation(self):
        '''This involves generating the mu and kappa arrays
        Then initialising based on self.start using k-means, EM-GMM or
        giving every instance a random probability of belonging to each cluster
        SH: 7June2013
        '''
        self.mu_list = np.ones((self.n_clusters,self.n_dimensions),dtype=float)
        self.std_list = np.ones((self.n_clusters,self.n_dimensions),dtype=float)
        self.LL_list = []
        self.zij = np.zeros((self.instance_array.shape[0],self.n_clusters),dtype=float)
        if self.start=='k_means':
            print 'Initialising clusters using a fast k_means run'
            self.cluster_assignments, self.cluster_details = k_means_clustering(self.instance_array, n_clusters=self.n_clusters, sin_cos = 1, number_of_starts = 3, seed=self.seed)
            for i in list(set(self.cluster_assignments)):
                self.zij[self.cluster_assignments==i,i] = 1
            print 'finished initialising'
        elif self.start=='EM_GMM':
            self.cluster_assignments, self.cluster_details = EM_GMM_clustering(self.instance_array, n_clusters=self.n_clusters, sin_cos = 1, number_of_starts = 1)
            for i in list(set(cluster_assignments)):
                self.zij[cluster_assignments==i,i] = 1
        else:
            print 'going with random option'
            #need to get this to work better.....
            self.zij = np.random.random(self.zij.shape)
            #and normalise so each row adds up to 1....
            self.zij = self.zij / ((np.sum(self.zij,axis=1))[:,np.newaxis])
        self._EM_GMM_maximisation_step()

    def _EM_GMM_maximisation_step(self):
        '''Maximisation step SH : 7June2013
        '''
        self.pi_hat = np.sum(self.zij,axis=0)/float(self.n_instances)
        self.mu_list_old = self.mu_list.copy()
        self.std_list_old = self.std_list.copy()
        for cluster_ident in range(self.n_clusters):
            #inst_tmp = (self.instance_array.T * self.zij[:,cluster_ident]).T
            #N= np.sum(self.zij[:,cluster_ident])
            #calculate the best fit for this cluster - all dimensions at once.... using new approximations
            self.std_list[cluster_ident,:], self.mu_list[cluster_ident,:] = EM_GMM_calc_best_fit(self.instance_array, self.zij[:,cluster_ident])
        #Prevent ridiculous situations happening....
        self.std_list = np.clip(self.std_list,0.001,300)
        self._EM_VMM_check_convergence()

    def _EM_VMM_check_convergence(self,):
        self.convergence_mu = np.sqrt(np.sum((self.mu_list_old - self.mu_list)**2))
        self.convergence_kappa = np.sqrt(np.sum((self.std_list_old - self.std_list)**2))

    def _EM_GMM_expectation_step(self,):
        self.probs = self.zij*0#np.ones((self.instance_array.shape[0],self.n_clusters),dtype=float)
        #instance_array_c and instance_array_s are used to speed up cos(instance_array - mu) using
        #the trig identity cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
        #this removes the need to constantly recalculate cos(a) and sin(a)
        for mu_tmp, std_tmp, p_hat, cluster_ident in zip(self.mu_list,self.std_list,self.pi_hat,range(self.n_clusters)):
            #norm_fac_exp = self.n_clusters*np.log(1./(2.*np.pi)) - np.sum(np.log(spec.iv(0,kappa_tmp)))
            #norm_fac_exp = -self.n_dimensions*np.log(2.*np.pi) - np.sum(np.log(spec.iv(0,std_tmp)))
            norm_fac_exp = self.n_dimensions*np.log(1./np.sqrt(2.*np.pi)) + np.sum(np.log(1./std_tmp))
            #pt1 = kappa_tmp * (self.instance_array_c*np.cos(mu_tmp) + self.instance_array_s*np.sin(mu_tmp))
            pt1 = -(self.instance_array - mu_tmp)**2/(2*(std_tmp**2))
            self.probs[:,cluster_ident] = p_hat * np.exp(np.sum(pt1,axis=1) + norm_fac_exp)
        prob_sum = (np.sum(self.probs,axis=1))[:,np.newaxis]
        self.zij = self.probs/(prob_sum)
        #Calculate the log-likelihood - note this is quite an expensive computation and not really necessary
        #unless comparing different techniques and/or checking for convergence
        #This is to prevent problems with log of a very small number....
        #L = np.sum(zij[probs>1.e-20]*np.log(probs[probs>1.e-20]))
        #L = np.sum(zij*np.log(np.clip(probs,1.e-10,1)))



#############################################################################
#####################Plotting functions#####################################
def plot_alfven_lines_func(ax):
    pickled_theoretical = pickle.load(file('theor_value.pickle','r'))
    mu = 4.*np.pi*(10**(-7))
    mi = 1.673*(10**(-27))
    meff = 2.5
    lamda = 0.27
    colors = ['k--','b-','r','y','g']
    for plotting_mode in range(0,2):
        B = 0.5
        va  = B/np.sqrt((10.**18) * meff * mi * mu)*lamda
        #ax.plot(pickled_theoretical['kh_list'],np.array(pickled_theoretical['k_par'][plotting_mode])*va/(2.*np.pi),'c-',linewidth=5)
        ax.plot(pickled_theoretical['kh_list'],np.array(pickled_theoretical['k_par'][plotting_mode])*va/(2.*np.pi),colors[plotting_mode],linewidth=4)
        #B = 0.5*5/7
        #va  = B/np.sqrt((10.**18) * meff * mi * mu)*lamda
        #ax.plot(pickled_theoretical['kh_list'],np.array(pickled_theoretical['k_par'][plotting_mode])*va/(2.*np.pi),'--')

def test_von_mises_fits():
    N = 20000
    mu = 2.9
    kappa = 5
    mu_list = np.linspace(-np.pi,np.pi,20)
    kappa_list = range(1,50)
    mu_best_fit = []
    mu_record = []
    fig,ax = pt.subplots(nrows = 2)
    def kappa_guess_func(kappa,R_e):
        return (R_e - spec.iv(1,kappa)/spec.iv(0,kappa))**2

    def calc_best_fit(theta):
        z = np.exp(1j*theta)
        z_bar = np.average(z)
        mean_theta = np.angle(z_bar)
        R_squared = np.real(z_bar * z_bar.conj())
        R_e = np.sqrt((float(N)/(float(N)-1))*(R_squared-1./float(N)))
        tmp1 = opt.fmin(kappa_guess_func,3,args=(R_e,))
        return mean_theta, tmp1[0]

    for mu in mu_list:
        kappa_best_fit = []
        for kappa in kappa_list:
            mu_record.append(mu)
            theta = np.random.vonmises(mu,kappa,N)
            mean_theta, kappa_best = calc_best_fit(theta)
            mu_best_fit.append(mean_theta)
            kappa_best_fit.append(kappa_best)
        ax[0].plot(kappa_list, kappa_best_fit,'.')
        ax[0].plot(kappa_list, kappa_list,'-')
    ax[1].plot(mu_record,mu_best_fit,'.')
    ax[1].plot(mu_record,mu_record,'-')
    fig.canvas.draw(); fig.show()

def generate_artificial_covar_data(n_clusters, n_dimensions, n_instances, prob=None, means=None, variances=None, covars=None, random_means_bounds = [-np.pi,np.pi], random_var_bounds = [0.0,0.02], random_covar_bounds = [0.0,0.1]):
    
    if prob==None:
        prob = np.ones((n_clusters,),dtype=float)*1./(n_clusters)
        print prob
    elif np.abs(np.sum(prob)-1)>0.001:
        raise ValueError('cluster probabilities dont add up to one within 0.001 tolerance....')
    n_instances_per_clust = np.array(map(int, (np.array(prob)*n_instances)))
    input_data = np.zeros((n_instances, n_dimensions),dtype=float)
    cluster_assignments = np.zeros((n_instances,),dtype=int)

    start_point = 0; end_point = 0
    #for i_ind in range(cur_covar.shape)
    if means==None:
        means = np.random.rand(n_clusters,n_dimensions)*(random_means_bounds[1]-random_means_bounds[0]) + random_means_bounds[0]

    covar_list = []
    variance_list = []
    #random entries, replace diagonals, make symmetric
    if covars==None:
        for i in range(n_clusters):
            covars = np.random.rand(n_dimensions, n_dimensions)*(random_covar_bounds[1]-random_covar_bounds[0]) + random_covar_bounds[0]
            covar_list.append(covars)

    if variances==None:
        variances = np.random.rand(n_clusters, n_dimensions)*(random_var_bounds[1]-random_var_bounds[0]) + random_var_bounds[0]
        for i in range(variances.shape[0]):
            np.fill_diagonal(covar_list[i], variances[i,:])

    for cur_covar in covar_list:
        for i in range(n_dimensions):
            for j in range(i+1,n_dimensions):
                cur_covar[j,i] = +cur_covar[i,j]

    #Make the covariance matrix definitive positive by making the diagonal
    #elements greater than the sum of the absolute values on each row
    #http://math.stackexchange.com/questions/332456/how-to-make-a-matrix-positive-semidefinite
    for index1 in range(len(covar_list)):
        covar = covar_list[index1]
        try:
            np.linalg.cholesky(covar)
        except np.linalg.LinAlgError:
            for i in range(covar.shape[0]):
                covar[i,i] = (np.sum(np.abs(covar[i,:])) - covar[i,i])*1.1
                variances[index1,i] = +covar[i,i]
            print 'problem'
            np.linalg.cholesky(covar)


    fig, ax = pt.subplots(nrows = 4, ncols=4); ax = ax.flatten()
    for loc, i in enumerate(covar_list): 
        im = ax[loc].imshow(np.abs(i), interpolation = 'nearest', cmap='binary')
        im.set_clim([0,0.2])
    fig.canvas.draw(); fig.show()

    fig, ax = pt.subplots(nrows = 4, ncols=4); ax = ax.flatten()
    for i in range(n_clusters):
        end_point = end_point + n_instances_per_clust[i]
        cur_mean = means[i,:]
        cur_covar = covar_list[i]
        np.linalg.cholesky(cur_covar)
        tmp_instances = n_instances_per_clust[i]
        print tmp_instances
        tmp = np.random.multivariate_normal(cur_mean, cur_covar, int(tmp_instances))
        input_data[start_point:end_point,:] = tmp
        cluster_assignments[start_point:end_point] = i
        start_point = end_point
        print tmp.shape
        im = ax[i].imshow(np.abs(np.cov(tmp.T)), interpolation = 'nearest', cmap='binary')
        im.set_clim([0,.2])

    fig.canvas.draw(); fig.show()
    input_data = input_data %(2.*np.pi)    
    input_data[input_data>np.pi] -= 2.*np.pi

    #shuffle the rows of the data so they
    #aren't in order of the clusters which might cause some problems....
    locs = np.arange(n_instances)
    np.random.shuffle(locs)
    input_data = input_data[locs,:]
    cluster_assignments = cluster_assignments[locs]
    feat_obj = feature_object(instance_array=input_data, misc_data_dict={})#, misc_data_labels):
    
    #create a clustering object with all info, and put it as the first object in the
    #clustered items list
    tmp = clustering_object()
    tmp.settings = {'method':'EM_VMM'}
    tmp.cluster_assignments = cluster_assignments
    tmp.cluster_details = {'EM_VMM_means':means,'EM_VMM_kappas':variances}
    tmp.feature_obj = feat_obj
    feat_obj.clustered_objects.append(tmp)
    return feat_obj




def generate_artificial_data(n_clusters, n_dimensions, n_instances, prob=None, method='vonMises', means=None, variances=None, random_means_bounds = [-np.pi,np.pi], random_var_bounds = [0.05,5]):
    '''Generate a dummy data set n_clusters : number of separate
    clusters n_dimensions : how many seperate phase signals per
    instance n_instances : number of instances - note this might be
    changed slightly depending on the probabilities

    kwargs prob : 1D array, length n_clusters, probabilty of each
    cluster - (note must add to one...)  if None, then all clusters
    will have equal probability method : distribution to draw points
    from - vonMises or Gaussian means, variances: arrays (n_clusters x
    n_dimensions) of the means and variances (kappa for vonMises) for
    the distributions if these are given, n_clusters and n_dimensions
    are ignored If only means or variances are given, then the missing
    one will be given random values Note for vonMises, 1/var is used
    as this is approximately kappa

    SH : 14May2013 '''
    if means != None and variances != None:
        if means.shape != variances.shape:
            raise ValueError('means and variances have different shapes')
        n_clusters, n_dimensions = means.shape
    if means!=None: n_clusters, n_dimensions = means.shape
    if variances!=None: n_clusters, n_dimensions = means.shape

    #randomly generate the means and variances using uniform distribution and the bounds given
    if means==None:
        means = np.random.rand(n_clusters,n_dimensions)*(random_means_bounds[1]-random_means_bounds[0]) + random_means_bounds[0]
    if variances==None:
        variances = np.random.rand(n_clusters,n_dimensions)*(random_var_bounds[1]-random_var_bounds[0]) + random_var_bounds[0]
    print means
    print variances
    #figure out how many instances per cluster based on the probabilities
    if prob==None:
        prob = np.ones((n_clusters,),dtype=float)*1./(n_clusters)
    elif np.abs(np.sum(prob)-1)>0.001:
        raise ValueError('cluster probabilities dont add up to one within 0.001 tolerance....')
    n_instances_per_clust = np.array(map(int, (np.array(prob)*n_instances)))
    input_data = np.zeros((n_instances, n_dimensions),dtype=float)
    cluster_assignments = np.zeros((n_instances,),dtype=int)

    start_point = 0; end_point = 0
    for i in range(n_clusters):
        end_point = end_point + n_instances_per_clust[i]
        for j in range(self.n_dimensions):
            #input_data[start_point:end_point,j] = vonmises.rvs(variances[i,j],size=n_instances_per_clust[i],loc=means[i,j],scale=1)
            if method=='vonMises':
                input_data[start_point:end_point,j] = vonmises.rvs(variances[i,j],size=n_instances_per_clust[i],loc=means[i,j],scale=1)
            elif method=='Gaussian':
                #input_data[start_point:end_point,j] = norm.rvs(size=n_instances_per_clust[i],loc=means[i,j],scale=1./variances[i,j]*3)
                sigma_converted = np.sqrt(-2*np.log(spec.iv(1,variances[i,j])/spec.iv(0,variances[i,j])))
                print sigma_converted
                input_data[start_point:end_point,j] = np.random.normal(size = n_instances_per_clust[i], loc = means[i,j], scale = sigma_converted)
                #input_data[start_point:end_point,j] = norm.rvs(size=n_instances_per_clust[i],loc=means[i,j],scale=1./variances[i,j]*3)
            else:
                raise ValueError('method must be either vonMises or Gaussian')
        cluster_assignments[start_point:end_point] = i
        start_point = end_point

    #Move data into [-pi,pi]
    input_data = input_data %(2.*np.pi)    
    input_data[input_data>np.pi] -= 2.*np.pi

    #shuffle the rows of the data so they
    #aren't in order of the clusters which might cause some problems....
    locs = np.arange(n_instances)
    np.random.shuffle(locs)
    input_data = input_data[locs,:]
    cluster_assignments = cluster_assignments[locs]
    feat_obj = feature_object(instance_array=input_data, misc_data_dict={})#, misc_data_labels):
    
    #create a clustering object with all info, and put it as the first object in the
    #clustered items list
    tmp = clustering_object()
    tmp.settings = {'method':'EM_VMM'}
    tmp.cluster_assignments = cluster_assignments
    tmp.cluster_details = {'EM_VMM_means':means,'EM_VMM_kappas':variances}
    tmp.feature_obj = feat_obj
    feat_obj.clustered_objects.append(tmp)
    return feat_obj

def make_grid_subplots(n_subplots, sharex = True, sharey = True, return_unflattened = False):
    '''This helper function generates the many subplots
    on a regular grid

    SH: 23May2013
    '''
    n_cols = int(math.ceil(n_subplots**0.5))
    if n_subplots/float(n_cols)>n_subplots/n_cols:
        n_rows = n_subplots/n_cols + 1
    else:
        n_rows = n_subplots/n_cols
    fig, ax = pt.subplots(nrows = n_rows, ncols = n_cols, sharex = True, sharey = True); ax_flat = ax.flatten()
    #fig, ax = pt.subplots(nrows = n_rows, ncols = n_cols,subplot_kw=dict(projection='polar')); ax = ax.flatten()
    if not return_unflattened:
        return fig, ax_flat
    else:
        return fig, ax_flat, ax

def modtwopi(x, offset=np.pi):
    """ return an angle in the range of offset +-
    2pi>>> print("{0:.3f}".format(modtwopi( 7),offset=3.14))0.717
    This simple strategy works when the number is near zero +- 2Npi,
    which is true for calculating the deviation from the cluster centre.
    does not attempt to make jumps small (use fix2pi_skips for that)
    """
    return ((-offset+np.pi+np.array(x)) % (2*np.pi) +offset -np.pi)

def convert_kappa_std(kappa,deg=True):
    '''This converts kappa from the von Mises distribution into a
    standard deviation that can be used to generate a similar normal
    distribution

    SH: 14June2013
    '''
    R_bar = spec.iv(1,kappa)/spec.iv(0,kappa)
    if deg==True:
        return np.sqrt(-2*np.log(R_bar))*180./np.pi
    else:
        return np.sqrt(-2*np.log(R_bar))

def EM_VMM_GMM_clustering_wrapper2(input_data):
    tmp = EM_VMM_GMM_clustering_class(*input_data)
    return copy.deepcopy(tmp.cluster_assignments), copy.deepcopy(tmp.cluster_details)

def EM_VMM_GMM_clustering_wrapper(instance_array, instance_array_amps, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random', kappa_calc='approx', hard_assignments = 0, kappa_converged = 0.1, mu_converged = 0.01, min_iterations=10, LL_converged = 1.e-4, verbose = 0, number_of_starts = 1):

    cluster_list = [n_clusters for i in range(number_of_starts)]
    seed_list = [i for i in range(number_of_starts)]
    rep = itertools.repeat
    from multiprocessing import Pool
    input_data_iter = itertools.izip(rep(instance_array), rep(instance_array_amps), rep(n_clusters),
                                     rep(n_iterations), rep(n_cpus), rep(start), rep(kappa_calc),
                                     rep(hard_assignments), rep(kappa_converged),
                                     rep(mu_converged),rep(min_iterations), rep(LL_converged),
                                     rep(verbose), seed_list)
    if n_cpus>1:
        pool_size = n_cpus
        pool = Pool(processes=pool_size, maxtasksperchild=3)
        print 'creating pool map'
        results = pool.map(EM_VMM_GMM_clustering_wrapper2, input_data_iter)
        print 'waiting for pool to close '
        pool.close()
        print 'joining pool'
        pool.join()
        print 'pool finished'
    else:
        results = map(EM_VMM_GMM_clustering_wrapper2, input_data_iter)
    LL_results = []
    for tmp in results: LL_results.append(tmp[1]['LL'][-1])
    print LL_results
    tmp_loc = np.argmax(LL_results)
    return results[tmp_loc]


def EM_GMM_GMM2_clustering_wrapper2(input_data):
    tmp = EM_GMM_GMM_clustering_class_self(*input_data)
    return copy.deepcopy(tmp.cluster_assignments), copy.deepcopy(tmp.cluster_details)

def EM_GMM_GMM2_clustering_wrapper(instance_array_amps, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random', kappa_calc='approx', hard_assignments = 0, kappa_converged = 0.1, mu_converged = 0.01, min_iterations=10, LL_converged = 1.e-4, verbose = 0, number_of_starts = 1):

    cluster_list = [n_clusters for i in range(number_of_starts)]
    seed_list = [i for i in range(number_of_starts)]
    rep = itertools.repeat
    from multiprocessing import Pool
    input_data_iter = itertools.izip(rep(instance_array_amps), rep(n_clusters),
                                     rep(n_iterations), rep(n_cpus), rep(start), rep(kappa_calc),
                                     rep(hard_assignments), rep(kappa_converged),
                                     rep(mu_converged),rep(min_iterations), rep(LL_converged),
                                     rep(verbose), seed_list)
    if n_cpus>1:
        pool_size = n_cpus
        pool = Pool(processes=pool_size, maxtasksperchild=3)
        print 'creating pool map'
        results = pool.map(EM_GMM_GMM2_clustering_wrapper2, input_data_iter)
        print 'waiting for pool to close '
        pool.close()
        print 'joining pool'
        pool.join()
        print 'pool finished'
    else:
        results = map(EM_GMM_GMM2_clustering_wrapper2, input_data_iter)
    LL_results = []
    for tmp in results: 
        print tmp[1]['LL'][-1]
        if tmp[1]['LL'][-1]!=0:
            LL_results.append(tmp[1]['LL'][-1])
        else:
            LL_results.append(-1.e20)
    print LL_results
    tmp_loc = np.argmax(LL_results)
    return results[tmp_loc]


class EM_VMM_GMM_clustering_class(clustering_object):
    def __init__(self, instance_array, instance_array_amps, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random', kappa_calc='approx', hard_assignments = 0, kappa_converged = 0.1, mu_converged = 0.01, min_iterations=10, LL_converged = 1.e-4, verbose = 0, seed=None):
        '''This model is supposed to include a mixture of Gaussian and von
        Mises distributions to allow datamining of data that essentially
        consists of complex numbers (amplitude and phase) such as most
        Fourier based measurements. Supposed to be an improvement on the
        case of just using the phases between channels - more interested
        in complex modes such as HAE, and also looking at data that is
        more amplitude based such as line of sight chord through the
        plasma for interferometers and imaging diagnostics.

        Note the amplitude data is included in
        misc_data_dict['mirnov_data'] from the stft-clustering
        extraction technique

        Need to figure out a way to normalise it... so that shapes of
        different amplitudes will look the same
        Need to plumb this in somehow...

        SH: 15June2013
        '''
        print 'hello'
        self.settings = {'n_clusters':n_clusters,'n_iterations':n_iterations,'n_cpus':n_cpus,'start':start,
                         'kappa_calc':kappa_calc,'hard_assignments':hard_assignments, 'method':'EM_VMM_GMM'}
        self.instance_array = copy.deepcopy(instance_array)
        #self.instance_array_amps = np.abs(instance_array_amps)
        #norm_factor = np.sum(np.abs(self.instance_array_amps),axis=1)
        #self.instance_array_amps = self.instance_array_amps/norm_factor[:,np.newaxis]
        tmp = np.zeros((instance_array_amps.shape[0], instance_array_amps.shape[1]-1),dtype=complex)
        for i in range(1,instance_array_amps.shape[1]):
            tmp[:,i-1] = instance_array_amps[:,i]/instance_array_amps[:,i-1]
        self.instance_array_amps = np.abs(tmp)
        self.instance_array_amps[np.angle(tmp)<0]*=(-1)
        print np.sum(self.instance_array_amps<0), np.sum(self.instance_array_amps>=0)
        self.instance_array_complex = np.exp(1j*self.instance_array)
        self.instance_array_c = np.real(self.instance_array_complex)
        self.instance_array_s = np.imag(self.instance_array_complex)
        self.n_instances, self.n_dimensions = self.instance_array.shape
        self.n_dimensions_amps = self.instance_array_amps.shape[1]
        self.n_clusters = n_clusters
        self.max_iterations = n_iterations
        self.start = start
        self.hard_assignments = hard_assignments
        self.seed = seed
        if self.seed == None:
            self.seed = os.getpid()
        print('seed,',self.seed)
        np.random.seed(self.seed)
        if kappa_calc == 'lookup_table':
            self.generate_bessel_lookup_table()
        else:
            self.bessel_lookup_table=None
        self.iteration = 1
        self.initialisation()
        self.convergence_record = []
        converged = 0 
        self.LL_diff = np.inf
        while converged!=1:
            start_time = time.time()
            self._EM_VMM_GMM_expectation_step()
            if self.hard_assignments:
                print 'hard assignments'
                self.cluster_assignments = np.argmax(self.zij,axis=1)
                self.zij = self.zij *0
                for i in range(self.n_clusters):
                    self.zij[self.cluster_assignments==i,i] = 1

            valid_items = self.probs>(1.e-300)
            self.LL_list.append(np.sum(self.zij[valid_items]*np.log(self.probs[valid_items])))
            self._EM_VMM_GMM_maximisation_step()
            if (self.iteration>=2): self.LL_diff = np.abs(((self.LL_list[-1] - self.LL_list[-2])/self.LL_list[-2]))
            if verbose:
                print 'Time for iteration %d :%.2f, mu_convergence:%.3e, kappa_convergence:%.3e, LL: %.8e, LL_dif : %.3e'%(self.iteration,time.time() - start_time,self.convergence_mu, self.convergence_kappa,self.LL_list[-1],self.LL_diff)
            self.convergence_record.append([self.iteration, self.convergence_mu, self.convergence_kappa])
            if (self.iteration > min_iterations) and (self.convergence_mu<mu_converged) and (self.convergence_kappa<kappa_converged) and (self.LL_diff<LL_converged):
                converged = 1
                print 'Convergence criteria met!!'
            elif self.iteration > n_iterations:
                converged = 1
                print 'Max number of iterations'
            self.iteration+=1
        print os.getpid(), 'Time for iteration %d :%.2f, mu_convergence:%.3e, kappa_convergence:%.3e, LL: %.8e, LL_dif : %.3e'%(self.iteration,time.time() - start_time,self.convergence_mu, self.convergence_kappa,self.LL_list[-1],self.LL_diff)
        #print 'AIC : %.2f'%(2*(mu_list.shape[0]*mu_list.shape[1])-2.*LL_list[-1])
        self.cluster_assignments = np.argmax(self.zij,axis=1)
        self.BIC = -2*self.LL_list[-1]+self.n_clusters*3*np.log(self.n_dimensions)
        #self.cluster_details = {'EM_VMM_means':self.mu_list, 'EM_VMM_kappas':self.kappa_list, 'LL':self.LL_list, 'zij':self.zij, 'BIC':self.BIC}
        self.cluster_details = {'EM_VMM_means':self.mu_list, 'EM_VMM_kappas':self.kappa_list, 'EM_GMM_means':self.mean_list, 'EM_GMM_std':self.std_list, 'LL':self.LL_list, 'BIC':self.BIC}

    def generate_bessel_lookup_table(self):
        self.kappa_lookup = np.linspace(0,100,10000)
        self.bessel_lookup_table = [spec.iv(1,kappa_lookup)/spec.iv(0,kappa_lookup), kappa_lookup]

    def initialisation(self):
        '''This involves generating the mu and kappa arrays
        Then initialising based on self.start using k-means, EM-GMM or
        giving every instance a random probability of belonging to each cluster
        SH: 7June2013
        '''
        self.mu_list = np.ones((self.n_clusters,self.n_dimensions),dtype=float)
        self.mean_list = np.ones((self.n_clusters,self.n_dimensions_amps),dtype=float)
        self.kappa_list = np.ones((self.n_clusters,self.n_dimensions),dtype=float)
        self.std_list = np.ones((self.n_clusters,self.n_dimensions_amps),dtype=float)
        self.LL_list = []
        self.zij = np.zeros((self.instance_array.shape[0],self.n_clusters),dtype=float)
        #maybe only the random option is valid here.....
        if self.start=='k_means':
            print 'Initialising clusters using a fast k_means run'
            self.cluster_assignments, self.cluster_details = k_means_clustering(self.instance_array, n_clusters=self.n_clusters, sin_cos = 1, number_of_starts = 3, seed=self.seed)
            for i in list(set(self.cluster_assignments)):
                self.zij[self.cluster_assignments==i,i] = 1
            print 'finished initialising'
        elif self.start=='EM_GMM':
            self.cluster_assignments, self.cluster_details = EM_GMM_clustering(self.instance_array, n_clusters=self.n_clusters, sin_cos = 1, number_of_starts = 1)
            for i in list(set(self.cluster_assignments)):
                self.zij[self.cluster_assignments==i,i] = 1
        else:
            print 'going with random option'
            #need to get this to work better.....
            self.zij = np.random.random(self.zij.shape)
            #and normalise so each row adds up to 1....
            self.zij = self.zij / ((np.sum(self.zij,axis=1))[:,np.newaxis])
        self._EM_VMM_GMM_maximisation_step()

    def _EM_VMM_GMM_maximisation_step(self):
        '''Maximisation step SH : 7June2013
        '''
        self.pi_hat = np.sum(self.zij,axis=0)/float(self.n_instances)
        self.mu_list_old = self.mu_list.copy()
        self.kappa_list_old = self.kappa_list.copy()
        self.mean_list_old = self.mean_list.copy()
        self.std_list_old = self.std_list.copy()

        for cluster_ident in range(self.n_clusters):
            inst_tmp = (self.instance_array_complex.T * self.zij[:,cluster_ident]).T
            N= np.sum(self.zij[:,cluster_ident])
            #calculate the best fit for this cluster - all dimensions at once.... using new approximations
            #VMM part
            self.kappa_list[cluster_ident,:], self.mu_list[cluster_ident,:], scale_fit1 = EM_VMM_calc_best_fit(inst_tmp, lookup=self.bessel_lookup_table, N=N)
            #GMM part
            self.std_list[cluster_ident,:], self.mean_list[cluster_ident,:] = EM_GMM_calc_best_fit(self.instance_array_amps, self.zij[:,cluster_ident])
        #Prevent ridiculous situations happening....
        self.kappa_list = np.clip(self.kappa_list,0.1,300)
        self.std_list = np.clip(self.std_list,0.001,300)
        self._EM_VMM_GMM_check_convergence()

    def _EM_VMM_GMM_check_convergence(self,):
        self.convergence_mu = np.sqrt(np.sum((self.mu_list_old - self.mu_list)**2))
        self.convergence_kappa = np.sqrt(np.sum((self.kappa_list_old - self.kappa_list)**2))
        self.convergence_mean = np.sqrt(np.sum((self.mean_list_old - self.mean_list)**2))
        self.convergence_std = np.sqrt(np.sum((self.std_list_old - self.std_list)**2))

    def _EM_VMM_GMM_expectation_step(self,):

        #Can probably remove this and modify zij directly
        self.probs = self.zij*0
        #instance_array_c and instance_array_s are used to speed up cos(instance_array - mu) using
        #the trig identity cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
        #this removes the need to constantly recalculate cos(a) and sin(a)
        for mu_tmp, kappa_tmp, mean_tmp, std_tmp, p_hat, cluster_ident in zip(self.mu_list,self.kappa_list,self.mean_list, self.std_list, self.pi_hat,range(self.n_clusters)):
            #norm_fac_exp = self.n_clusters*np.log(1./(2.*np.pi)) - np.sum(np.log(spec.iv(0,kappa_tmp)))
            norm_fac_exp_VMM = -self.n_dimensions*np.log(2.*np.pi) - np.sum(np.log(spec.iv(0,kappa_tmp)))
            norm_fac_exp_GMM = -self.n_dimensions_amps*np.log(2.*np.pi) - np.sum(np.log(std_tmp))
            pt1_GMM = -(self.instance_array_amps - mean_tmp)**2/(2*(std_tmp**2))
            #Note the use of instance_array_c and s is from a trig identity
            #This speeds this calc up significantly because mu_tmp is small compared to instance_array_c and s
            #which can be precalculated
            pt1_VMM = kappa_tmp * (self.instance_array_c*np.cos(mu_tmp) + self.instance_array_s*np.sin(mu_tmp))
            self.probs[:,cluster_ident] = p_hat * np.exp(np.sum(pt1_VMM,axis=1) + norm_fac_exp_VMM +
                                                         np.sum(pt1_GMM,axis=1) + norm_fac_exp_GMM)
        #This is to make sure the row sum is 1
        prob_sum = (np.sum(self.probs,axis=1))[:,np.newaxis]
        self.zij = self.probs/(prob_sum)
        #Calculate the log-likelihood - note this is quite an expensive computation and not really necessary
        #unless comparing different techniques and/or checking for convergence
        #This is to prevent problems with log of a very small number....
        #L = np.sum(zij[probs>1.e-20]*np.log(probs[probs>1.e-20]))
        #L = np.sum(zij*np.log(np.clip(probs,1.e-10,1)))

def norm_bet_chans(input_data, method = 'sum', reference = 0):
    '''
    How to normalise between channels for the clustering - i.e sum,
    adjacent etc... to remove the problems with different timings

    SRH: 24May2014
    '''
    if method == 'sum':
        output_data = input_data/np.mean(input_data,axis = 1)[:,np.newaxis]
    elif method == 'adj':
        instances, dims = input_data.shape
        output_data = np.ones((instances, dims - 1),dtype=complex)
        for i in range(1,dims): 
            output_data[:, i-1] = input_data[:,i]/input_data[:,i-1]
    elif method == 'ref':
        output_data = input_data / (input_data[:,0])[:,np.newaxis]
    elif method == 'adj-self':
        instances, dims = input_data.shape
        output_data = np.ones((instances, dims - 1),dtype=complex)
        for i in range(1,dims): 
            output_data[:, i-1] = input_data[i,:]/(input_data[:,i-1]+input_data[:,i])
    return output_data

class EM_GMM_GMM_clustering_class_self(clustering_object):
    '''
    This model is uses Gaussian mixtures for the real and imaginary components of the values

    Args:
      instance_array_amps (n_instances x n_dims, complex np.array): data used for clustering

    Kwargs:
      norm_method (str): any of sum, adj, ref, adj-self
      start (str): any of random, k-means, EM_GMM

    SH: 20May2014
    '''
    def __init__(self, instance_array_amps, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random', kappa_calc='approx', hard_assignments = 0, kappa_converged = 0.1, mu_converged = 0.01, min_iterations=10, LL_converged = 1.e-4, verbose = 0, seed=None, norm_method = 'sum'):
        print 'EM_GMM_GMM2', instance_array_amps.shape
        self.settings = {'n_clusters':n_clusters,'n_iterations':n_iterations,'n_cpus':n_cpus,'start':start,
                         'kappa_calc':kappa_calc,'hard_assignments':hard_assignments, 'method':'EM_VMM_GMM'}
        #self.instance_array = copy.deepcopy(instance_array)
        self.instance_array_amps = instance_array_amps
        self.data_complex = norm_bet_chans(instance_array_amps, method = norm_method)
        print 'hello norm method',  norm_method
        self.data_complex = instance_array_amps/np.sum(instance_array_amps,axis = 1)[:,np.newaxis]
        self.input_data = np.hstack((np.real(self.data_complex), np.imag(self.data_complex)))
        self.n_dim = self.data_complex.shape[1]
        self.n_instances, self.n_dimensions = self.input_data.shape

        self.n_clusters = n_clusters; self.max_iterations = n_iterations; self.start = start
        self.hard_assignments = hard_assignments; self.seed = seed
        if self.seed == None: self.seed = os.getpid()
        print('seed,',self.seed)
        np.random.seed(self.seed)
        self.iteration = 1

        self._initialisation()
        self.convergence_record = []; converged = 0 
        self.LL_diff = np.inf
        while converged!=1:
            start_time = time.time()
            self._EM_VMM_GMM_expectation_step()
            if self.hard_assignments:
                print 'hard assignments'
                self.cluster_assignments = np.argmax(self.zij,axis=1)
                self.zij = self.zij *0
                for i in range(self.n_clusters):
                    self.zij[self.cluster_assignments==i,i] = 1

            valid_items = self.probs>(1.e-300)
            self.LL_list.append(np.sum(self.zij[valid_items]*np.log(self.probs[valid_items])))
            self._EM_VMM_GMM_maximisation_step()
            if (self.iteration>=2): self.LL_diff = np.abs(((self.LL_list[-1] - self.LL_list[-2])/self.LL_list[-2]))
            if verbose:
                print 'Time for iteration %d :%.2f, mu_convergence:%.3e, kappa_convergence:%.3e, LL: %.8e, LL_dif : %.3e'%(self.iteration,time.time() - start_time,self.convergence_mean, self.convergence_std, self.LL_list[-1],self.LL_diff)
            self.convergence_record.append([self.iteration, self.convergence_mean, self.convergence_std])
            mean_converged = mu_converged; std_converged = kappa_converged
            if (self.iteration > min_iterations) and (self.convergence_mean<mean_converged) and (self.convergence_std<std_converged) and (self.LL_diff<LL_converged):
                converged = 1
                print 'Convergence criteria met!!'
            elif self.iteration > n_iterations:
                converged = 1
                print 'Max number of iterations'
            self.iteration+=1
        print os.getpid(), 'Time for iteration %d :%.2f, mu_convergence:%.3e, kappa_convergence:%.3e, LL: %.8e, LL_dif : %.3e'%(self.iteration,time.time() - start_time,self.convergence_mean, self.convergence_std,self.LL_list[-1],self.LL_diff)
        #print 'AIC : %.2f'%(2*(mu_list.shape[0]*mu_list.shape[1])-2.*LL_list[-1])
        self.cluster_assignments = np.argmax(self.zij,axis=1)
        self.BIC = -2*self.LL_list[-1]+self.n_clusters*3*np.log(self.n_dimensions)
        gmm_means_re, gmm_means_im = np.hsplit(self.mean_list, 2)
        gmm_vars_re, gmm_vars_im = np.hsplit(self.std_list**2, 2)

        self.cluster_details = {'EM_GMM_means':self.mean_list, 'EM_GMM_variances':self.std_list**2,'BIC':self.BIC,'LL':self.LL_list, 
                                'EM_GMM_means_re':gmm_means_re, 'EM_GMM_variances_re':gmm_vars_re,
                                'EM_GMM_means_im':gmm_means_im, 'EM_GMM_variances_im':gmm_vars_im}



    def _initialisation(self):
        '''This involves generating the mu and kappa arrays
        Then initialising based on self.start using k-means, EM-GMM or
        giving every instance a random probability of belonging to each cluster
        SH: 7June2013
        '''
        self.mean_list = np.ones((self.n_clusters,self.n_dimensions),dtype=float)
        self.std_list = np.ones((self.n_clusters,self.n_dimensions),dtype=float)
        self.LL_list = []
        self.zij = np.zeros((self.n_instances, self.n_clusters),dtype=float)
        #maybe only the random option is valid here.....
        if self.start=='k_means':
            print 'Initialising clusters using a fast k_means run'
            self.cluster_assignments, self.cluster_details = k_means_clustering(self.input_data, n_clusters=self.n_clusters, sin_cos = 0, number_of_starts = 4, seed=self.seed)
            for i in list(set(self.cluster_assignments)):
                self.zij[self.cluster_assignments==i,i] = 1
            #print 'finished initialising'
        elif self.start=='EM_GMM':
            self.cluster_assignments, self.cluster_details = EM_GMM_clustering(self.input_data, n_clusters=self.n_clusters, sin_cos = 1, number_of_starts = 1)
            for i in list(set(self.cluster_assignments)):
                self.zij[self.cluster_assignments==i,i] = 1
        else:
            print 'going with random option'
            #need to get this to work better.....
            self.zij = np.random.random(self.zij.shape)
            #and normalise so each row adds up to 1....
            self.zij = self.zij / ((np.sum(self.zij,axis=1))[:,np.newaxis])
        self._EM_VMM_GMM_maximisation_step()

    def _EM_VMM_GMM_maximisation_step(self):
        '''Maximisation step SH : 7June2013
        '''
        self.pi_hat = np.sum(self.zij,axis=0)/float(self.n_instances)
        self.mean_list_old = self.mean_list.copy()
        self.std_list_old = self.std_list.copy()
        for cluster_ident in range(self.n_clusters):
            self.std_list[cluster_ident,:], self.mean_list[cluster_ident,:] = EM_GMM_calc_best_fit(self.input_data, self.zij[:,cluster_ident])
        #Prevent ridiculous situations happening....
        #self.std_list = np.clip(self.std_list,0.5,300)
        self.std_list = np.clip(self.std_list,0.001,300)
        self._EM_VMM_GMM_check_convergence()

    def _EM_VMM_GMM_check_convergence(self,):
        self.convergence_mean = np.sqrt(np.sum((self.mean_list_old - self.mean_list)**2))
        self.convergence_std = np.sqrt(np.sum((self.std_list_old - self.std_list)**2))

    def _EM_VMM_GMM_expectation_step(self,):
        #Can probably remove this and modify zij directly
        self.probs = self.zij*0
        for mean_tmp, std_tmp, p_hat, cluster_ident in zip(self.mean_list, self.std_list, self.pi_hat,range(self.n_clusters)):
            norm_fac_exp_GMM = -self.n_dimensions*np.log(2.*np.pi) - np.sum(np.log(std_tmp))
            pt1_GMM = -(self.input_data - mean_tmp)**2/(2*(std_tmp**2))
            self.probs[:,cluster_ident] = p_hat * np.exp(np.sum(pt1_GMM,axis=1) + norm_fac_exp_GMM)
        #This is to make sure the row sum is 1
        prob_sum = (np.sum(self.probs,axis=1))[:,np.newaxis]
        self.zij = self.probs/(prob_sum)


def EM_GMM_GMM_clustering(instance_array_amps, n_clusters=9, sin_cos = 0, number_of_starts = 10, show_covariances = 0, clim=None, covariance_type='diag', n_iter = 50):
    '''
    Cluster using a Gaussian for the real and imag part of the ratio of the complex value between adjacent channels
    Supposed to be for imaging diagnostics

    SRH: 18May2014
    '''
    print 'starting EM-GMM-GMM algorithm from sckit-learn, clusters=%d, retries : %d'%(n_clusters,number_of_starts)
    #tmp = np.zeros((instance_array_amps.shape[0], instance_array_amps.shape[1]-1),dtype=complex)
    #for i in range(1,instance_array_amps.shape[1]):
    #    tmp[:,i-1] = instance_array_amps[:,i]/instance_array_amps[:,i-1]
    #print 'ratio :', np.sum(np.abs(np.imag(instance_array_amps)))/np.sum(np.abs(np.real(instance_array_amps)))
    data_complex = instance_array_amps/np.sum(instance_array_amps,axis = 1)[:,np.newaxis]
    #data_complex = instance_array_amps/(instance_array_amps[:,2])[:,np.newaxis]
    #print 'hello..', instance_array_amps.shape
    input_data = np.hstack((np.real(data_complex), np.real(data_complex)))
    #k_means_cluster_assignments, k_means_cluster_details = k_means_clustering(input_data, n_clusters=n_clusters, sin_cos = 1, number_of_starts = 3,)
    #print k_means_cluster_assignments
    #input_data = np.hstack((np.abs(data_complex),(np.abs(data_complex))))
    n_dim = data_complex.shape[1]
    #print n_clusters
    gmm = mixture.GMM(n_components = n_clusters, covariance_type = covariance_type, n_init = number_of_starts, n_iter = n_iter,)
    gmm.fit(input_data)
    cluster_assignments = gmm.predict(input_data)
    bic_value = gmm.bic(input_data)
    LL = np.sum(gmm.score(input_data))

    #Extract the means, variances and covariances
    gmm_covars = np.array(gmm._get_covars())
    gmm_vars = np.array([np.diagonal(i) for i in gmm._get_covars()])
    gmm_vars_re, gmm_vars_im = np.hsplit(gmm_vars,2)
    gmm_covars_re = np.array([i[0:n_dim,0:n_dim] for i in gmm._get_covars()])
    gmm_covars_im = np.array([i[n_dim:,n_dim:] for i in gmm._get_covars()])
    gmm_means = gmm.means_
    gmm_means_re, gmm_means_im = np.hsplit(gmm_means, 2)
    #Bundle up the answer
    cluster_details = {'EM_GMM_means':gmm_means, 'EM_GMM_variances':gmm_vars, 'EM_GMM_covariances':gmm_covars, 'EM_GMM_means_re':gmm_means_re, 'EM_GMM_variances_re':gmm_vars_re, 'EM_GMM_covariances_re':gmm_covars_re,'EM_GMM_means_im':gmm_means_im, 'EM_GMM_variances_im':gmm_vars_im, 'EM_GMM_covariances_im':gmm_covars_im,'BIC':bic_value,'LL':LL}
    print 'EM_GMM_GMM Converged: ', gmm.converged_

    return cluster_assignments, cluster_details
