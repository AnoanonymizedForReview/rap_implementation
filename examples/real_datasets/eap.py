from base_code.algos.rap import EAP
from base_code.analysis.metrics_base import MetricsBase, Result
import numpy as np


def calc_sap(**kwargs):
    pref = kwargs['pref']
    q = kwargs['q']
    S = kwargs['S']
    eps = kwargs.get('eps', np.max(S))
    MAX_IT = kwargs.get('max_it', 7000)
    CONV_IT = kwargs.get('conv_itr', 700)
    DAMPING = kwargs.get('damping', 0.99)
    gnd_cls = kwargs['gnd_cls']
    N_T = kwargs.get('N_T', 2)

    print("Evaluating preference = {:2.5f}, p2={:2.5f}".format(pref, q))
    af = EAP(**{
        'S': S,
        'preference': pref,
        'q': q,
        'eps': eps,
        'damping': DAMPING,
        'max_iter': MAX_IT,
        'convergence_iter': CONV_IT,
        'N_T': N_T
    }).run()

    cluster_centers_indices = af.decision.K
    exemplars_in_clusters = {
        k: [e for e in v if e in cluster_centers_indices]
        for k, v in af.decision.clusters.items()
    }

    ############################
    print("Exemplars: {}".format(len(cluster_centers_indices)))
    print("max<K_i>: {}".format(max([len(e) for e in exemplars_in_clusters.values()])))
    print("Clusters: {}".format(len(af.decision.clusters.keys())))
    results = MetricsBase(
        gnd_cls=gnd_cls, cls=af.decision.clusters, verbose=True).compute().results
    results.CONVERGED = af.converged
    print('converged' if af.converged else 'did not converge')
    results.EXTRA = {"q" : q, "p" : pref, 'iter':af.c_itr,
                     'max_K_i': max([len(e) for e in exemplars_in_clusters.values()])
    }
    kwargs['return'][str(pref)+'_'+str(q)] = results
