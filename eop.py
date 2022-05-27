"""
Taken from here and a bit modified
https://github.com/dodgejesse/show_your_work/
"""

import numpy as np
from typing import List, Dict, Union


def _cdf_with_replacement(i,n,N):
    return (i/N)**n


def _compute_stds(N, cur_data, expected_max_cond_n, pdfs):
    """
    this computes the standard error of the max.
    this is what the std dev of the bootstrap estimates of the mean of the max converges to, as
    is stated in the last sentence of the summary on page 10 of 
    http://www.stat.cmu.edu/~larry/=stat705/Lecture13.pdf
    """
    std_of_max_cond_n = []
    for n in range(N):
        # for a given n, estimate variance with \sum(p(x) * (x-mu)^2), where mu is \sum(p(x) * x).
        cur_std = 0
        for i in range(N):
            cur_std += (cur_data[i] - expected_max_cond_n[n])**2 * pdfs[n][i]
        cur_std = np.sqrt(cur_std)
        std_of_max_cond_n.append(cur_std)
    return std_of_max_cond_n
    

# this implementation assumes sampling with replacement for computing the empirical cdf
def expected_online_performance(
    online_performance: List[float],
    output_n          : int
) -> Dict[str, Union[List[float], float]]:
    # Copy and sort?
    online_performance = list(online_performance)
    online_performance.sort()

    N    = len(online_performance)
    pdfs = []
    for n in range(1,N+1):
        # the CDF of the max
        F_Y_of_y = []
        for i in range(1,N+1):
            F_Y_of_y.append(_cdf_with_replacement(i,n,N))


        f_Y_of_y = []
        cur_cdf_val = 0
        for i in range(len(F_Y_of_y)):
            f_Y_of_y.append(F_Y_of_y[i] - cur_cdf_val)
            cur_cdf_val = F_Y_of_y[i]
        
        pdfs.append(f_Y_of_y)

    expected_max_cond_n = []
    for n in range(N):
        # for a given n, estimate expected value with \sum(x * p(x)), where p(x) is prob x is max.
        cur_expected = 0
        for i in range(N):
            cur_expected += online_performance[i] * pdfs[n][i]
        expected_max_cond_n.append(cur_expected)


    std_of_max_cond_n = _compute_stds(N, online_performance, expected_max_cond_n, pdfs)

    return {
        "mean": expected_max_cond_n[:output_n],
        "std" : std_of_max_cond_n[:output_n],
        "max" : np.max(online_performance),
        "min" : np.min(online_performance)
    }

def expected_online_performance_arbit(
    online_performance : List[float],
    offline_performance: List[float],
    output_n           : int
) -> Dict[str, Union[List[float], float]]:
    means = [x for _, x in sorted(zip(offline_performance, online_performance), key=lambda pair: pair[0], reverse=True)]

    if len(means) > 0:
        cur_max = means[0]
        for ind in range(len(means)):
            cur_max = max(cur_max, means[ind])
            means[ind] = cur_max

    return {
        "mean": means[:output_n],
        "std" : means[:output_n],
        "max" : np.max(online_performance),
        "min" : np.min(online_performance)
    }


if __name__ == "__main__":
    example_valid_perf = np.random.uniform(0,1, 20)
    print(expected_online_performance(example_valid_perf))