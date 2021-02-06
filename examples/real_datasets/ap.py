from sklearn.cluster.affinity_propagation_ import AffinityPropagation
from base_code.analysis.metrics_base import MetricsBase, Result
import numpy as np


#for evaluating print output
import io
from contextlib import redirect_stdout

def calc_ap(**kwargs):
    pref = kwargs['pref']
    S = kwargs['S']
    MAX_IT = kwargs.get('max_it', 7000)
    CONV_IT = kwargs.get('conv_itr', 700)
    DAMPING = kwargs.get('damping', 0.99)
    gnd_cls = kwargs['gnd_cls']
    print("Evaluating preference = {:2.5f}".format(pref))

    # for evaluating print output
    with io.StringIO() as buf, redirect_stdout(buf):
        af = AffinityPropagation(
            affinity='precomputed', preference=pref, damping=DAMPING, max_iter=MAX_IT,
            convergence_iter=CONV_IT, verbose=True
        ).fit(S)
        output = buf.getvalue()
        if "not" in output:
            converged = False
        else:
            converged = True
    ############################

    print('converged' if converged else 'did not converge')
    c_centers = af.cluster_centers_indices_
    c_labels = af.labels_
    print('number of clusters:{}'.format(len(c_centers)))

    cls = {}
    for k in range(len(c_centers)):
        cls[k] = np.where(c_labels == k)[0]

    results = MetricsBase(gnd_cls=gnd_cls, cls=cls, verbose=True).compute().results # type: Result
    results.CONVERGED = converged
    results.EXTRA = {'p': pref, 'output':output}
    kwargs['return'][str(pref)] = results
