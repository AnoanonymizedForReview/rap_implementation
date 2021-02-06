from base_code.analysis.metrics_base import MetricsBase, Result
from base_code.analysis.overall_clusters import ChainAggregator, ChainAggregatorMethod

def calc_greedy(**kwargs):
    thr = kwargs['thr']
    S = kwargs['S']
    gnd_cls=kwargs['gnd_cls']
    print("Evaluating threshold = {:2.5f}".format(thr))
    H = S > thr
    cls = ChainAggregator(H=H, method=ChainAggregatorMethod.SYMMETRIC_H).get_clusters()


    results = MetricsBase(gnd_cls=gnd_cls, cls=cls, verbose=True).compute().results
    kwargs['return'][str(thr)] = results


