from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from base_code.analysis.overall_clusters import ChainAggregator, ChainAggregatorMethod
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib.pyplot import Figure, Axes


class InfoType(Enum):

    SCATTER_BASE_COLORED = 1 # colored plot of ground truth clusters
    SCATTER_BASE_GRAY = 2 # plot of ground truth clusters without colors
    SCATTER_CENTER_WITH_DENSITY = 3 # density plot for potential local exemplars
    SCATTER_CENTER_WITH_CLUSTER_TYPES = 4 # scatter plot of potential lcal exemplars along with colors to show cluster types
    SCATTER_CONFIDENCE = 5 # scatter plot of all points, with colors showing confidence. You might need to devise your own confidence function
    SCATTER_FINAL_WITH_EXEMPLARS = 6 # final scatter plot along with local exemplars
    SCATTER_FINAL_WITHOUT_EXEMPLARS = 7 # final scatter plot without local exemplars
    SCATTER_FINAL_WITH_LINKS = 8 # final scatter plot with edges
    HISTOGRAM_CONNECTIONS = 9 # histogram showing connections vs number of points for looking at relative densities
    HISTOGRAM_INTER_EXEMPLAR_LINKS = 10 # histogram of inter-exemplar links, to find out potential inconsistencies


class InfoCmaps:
    cmap_names = ['Reds', 'Greens', 'Blues', 'jet','YlOrBr', 'Greys',
        'Purples',  'Greys', 'Oranges', 'YlOrRd',
        'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'Reds', 'PuBuGn', 'BuGn', 'YlGn']

    def __init__(self):
        self.cmaps = np.array([plt.get_cmap(name) for name in InfoCmaps.cmap_names])
        self.cmap_samples = np.array([cm(0.9) for cm in self.cmaps])


class InfoPlotter:



    def __init__(self, **kwargs):
        self.info = kwargs['type'] # type: InfoType
        assert isinstance(self.info, InfoType)
        try:
            self.fig = kwargs['fig'] # type: Figure
            self.ax = kwargs['ax'] # type: Axes
        except:
            self.fig, self.ax = kwargs.get('plot', plt).subplots()

        self.state = np.random.RandomState(0)
        self.X = kwargs['X']
        self.H = kwargs.get('H', None)
        self.gnd_cls = kwargs.get('gnd_cls', None)
        if min(self.gnd_cls.keys()) > 0:
            self.gnd_cls = {k- min(self.gnd_cls.keys()):v for k,v in self.gnd_cls.items()}
        self.S = kwargs.get('S', None)
        self.cls = ChainAggregator(self.H).get_clusters() # type: dict
        self.cmaps = InfoCmaps()

    def __scatter_base_colored(self):
        inverted_gnd_cls = dict((v, k) for k in self.gnd_cls for v in self.gnd_cls[k])
        labels_true = [int(inverted_gnd_cls[k]) for k in sorted(inverted_gnd_cls.keys())]
        self.ax.scatter(self.X[:, 0], self.X[:, 1],
                        c=self.cmaps.cmap_samples[labels_true])

    def __scatter_base_gray(self):
        self.ax.scatter(self.X[:, 0], self.X[:, 1], c='#595959')


    def __scatter_centers_with_density(self):
        c_centers = np.where(np.diagonal(self.H) == 1)[0]
        center_connections = np.sum(self.H, axis=1)[c_centers]
        vmin = min(center_connections)
        vmax = max(center_connections)
        # cm = list(plt.cm.cmap_d.values())[self.state.randint(0,100)]
        cm = plt.get_cmap('Reds')
        self.ax.scatter(self.X[:, 0], self.X[:, 1], marker='x', c='#5F5F5F')
        sc = self.ax.scatter(
            self.X[c_centers, 0], self.X[c_centers, 1],
            c=center_connections, cmap=cm, vmin=vmin, vmax=vmax
        )
        cbaxes = inset_axes(self.ax, '80%', '5%', loc=9)
        self.fig.colorbar(sc, orientation='horizontal', cax=cbaxes, ticks=[vmin, vmax])

    def __scatter_center_with_cluster_types(self):
        c_centers = np.where(np.diagonal(self.H) == 1)[0]
        exemplars_in_clusters = {
            k: [e for e in v if e in c_centers]
            for k, v in self.cls.items()
        }
        inverted_cls = dict((v, k) for k in self.cls for v in self.cls[k])
        labels_pred = [inverted_cls[k] for k in sorted(inverted_cls.keys())]

        self.ax.scatter(self.X[:, 0], self.X[:, 1], marker='x', c='#5F5F5F')
        self.ax.scatter(self.X[c_centers, 0], self.X[c_centers, 1],
                        c=self.cmaps.cmap_samples[np.array(labels_pred)[c_centers]], linewidths=0)

    def __scatter_confidence(self):
        conf = np.zeros(self.H.shape[0])
        c_centers = np.where(np.diagonal(self.H) == 1)[0]
        exemplars_in_clusters = {
            k: [e for e in v if e in c_centers]
            for k, v in self.cls.items()
        }
        for k in self.cls.keys():
            cls_best = np.max(
                self.S[np.array(self.cls[k])[:, None], exemplars_in_clusters[k]], axis=1)
            non_cls_exemplars = [x for x in c_centers if x not in exemplars_in_clusters[k]]
            non_cls_best = np.max(
                self.S[np.array(self.cls[k])[:, None], non_cls_exemplars], axis=1)
            conf[self.cls[k]] = np.log(cls_best - non_cls_best)
        cm = plt.get_cmap('CMRmap')
        vmin = min(conf)
        vmax = max(conf)
        sc = self.ax.scatter(self.X[:, 0], self.X[:, 1],
                             c=conf, cmap=cm, vmin=vmin, vmax=vmax)
        cbaxes = inset_axes(self.ax, '80%', '5%', loc=9)
        self.fig.colorbar(sc, orientation='horizontal', cax=cbaxes, ticks=[vmin, vmax])

    def __histogram_inter_exemplar_links(self):
        total_str = []
        for c_idx, c_points in self.cls.items():
            c_centers = np.where(np.diagonal(self.H) == 1)[0]
            local_exemplars = [x for x in c_points if x in c_centers]

            strengh = np.zeros((len(local_exemplars), len(local_exemplars)))
            for i in range(len(local_exemplars)):
                for j in range(i + 1, len(local_exemplars)):
                    commons = \
                    np.where(np.bitwise_and(self.H[:, local_exemplars[i]], self.H[:, local_exemplars[j]]) == 1)[0]
                    strengh[i, j] = len(commons)
            total_str.extend(strengh.flatten().tolist())
        total_str = [x for x in total_str if x != 0]
        try:
            bins = [-0.5] + np.arange(np.min(total_str) - 0.5, max(5, np.max(total_str)) + 0.5).tolist() + [
                max(5, np.max(total_str)) + 0.5]
            self.ax.hist(
                total_str, bins=bins, facecolor='blue', alpha=0.75
            )
        except:
            print('no exemplars to compare for this dataset')

    def __histogram_connections(self):
        connections = self.H.sum(axis=1)
        bins = [-0.5] + np.arange(min(connections) - 0.5, max(5, max(connections)) + 0.5).tolist() + [
            max(5, max(connections)) + 0.5]
        conn = [[connections[v] for v in self.cls[k]] for k in sorted(list(self.cls.keys()))]
        self.ax.hist(
            conn,
            bins=bins, stacked=True, lw=0, color = self.cmaps.cmap_samples[sorted(list(set(self.cls.keys())))]
        )


    def __scatter_final_without_exemplars(self):
        inverted_cls = dict((v, k) for k in self.cls for v in self.cls[k])
        labels_pred = [inverted_cls[k] for k in sorted(inverted_cls.keys())]

        self.ax.scatter(self.X[:, 0], self.X[:, 1],
                            marker='x', c=self.cmaps.cmap_samples[labels_pred])

    def __scatter_final_with_exemplars(self):
        inverted_cls = dict((v, k) for k in self.cls for v in self.cls[k])
        labels_pred = [inverted_cls[k] for k in sorted(inverted_cls.keys())]
        c_centers = np.where(np.diagonal(self.H) == 1)[0]
        self.ax.scatter(self.X[:, 0], self.X[:, 1],
                        marker='x', c=self.cmaps.cmap_samples[labels_pred])
        self.ax.scatter(self.X[c_centers, 0], self.X[c_centers, 1],
                        c=self.cmaps.cmap_samples[np.array(labels_pred)[c_centers]])

    def __scatter_final_with_links(self):
        plot_centers = True
        c_centers = np.where(np.diagonal(self.H) == 1)[0]
        for cls_idx in range(len(self.cls.keys())):
            cls_exemplars = [x for x in c_centers if x in self.cls[cls_idx]]
            cmap = self.cmaps.cmaps[cls_idx]
            colors = cmap(np.linspace(0.7, 1, len(cls_exemplars)))

            for k, col in zip(range(len(cls_exemplars)), colors):
                class_members = np.where(self.H[:, cls_exemplars[k]] == 1)
                cluster_center = self.X[cls_exemplars[k]]
                self.ax.plot(self.X[class_members, 0][0], self.X[class_members, 1][0], 'x', c=col)
                for x in self.X[class_members]:
                    self.ax.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], c=col)
                    if plot_centers:
                        self.ax.plot(cluster_center[0], cluster_center[1], 'o', c=col,
                                      markeredgecolor='k', markersize=5)


    def compute(self):
        if self.info == InfoType.SCATTER_BASE_COLORED:
            self.__scatter_base_colored()
        elif self.info == InfoType.SCATTER_BASE_GRAY:
            self.__scatter_base_gray()
        elif self.info == InfoType.SCATTER_CENTER_WITH_DENSITY:
            self.__scatter_centers_with_density()
        elif self.info == InfoType.SCATTER_CENTER_WITH_CLUSTER_TYPES:
            self.__scatter_center_with_cluster_types()
        elif self.info == InfoType.SCATTER_CONFIDENCE:
            self.__scatter_confidence()
        elif self.info == InfoType.HISTOGRAM_CONNECTIONS:
            self.__histogram_connections()
        elif self.info == InfoType.HISTOGRAM_INTER_EXEMPLAR_LINKS:
            self.__histogram_inter_exemplar_links()
        elif self.info == InfoType.SCATTER_FINAL_WITH_EXEMPLARS:
            self.__scatter_final_with_exemplars()
        elif self.info == InfoType.SCATTER_FINAL_WITHOUT_EXEMPLARS:
            self.__scatter_final_without_exemplars()
        elif self.info == InfoType.SCATTER_FINAL_WITH_LINKS:
            self.__scatter_final_with_links()
            #TODO additional params in kwargs for plot titles etc
        return self


    # def title(self):
    #     self.ax.set_title(self.info.name)


