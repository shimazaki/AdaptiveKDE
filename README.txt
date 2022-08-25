This package implements adaptive kernel density estimation algorithms for 1-dimensional 
signals developed by Hideaki Shimazaki. This enables the generation of smoothed histograms
that preserve important density features at multiple scales, as opposed to naive
single-bandwidth kernel density methods that can either over or under smooth density
estimates. These methods are described in Shimazaki's paper:

 H. Shimazaki and S. Shinomoto, "Kernel Bandwidth Optimization in Spike Rate Estimation," 
 in Journal of Computational Neuroscience 29(1-2): 171–182, 2010 
 http://dx.doi.org/10.1007/s10827-009-0180-4.
 
License:
All software in this package is licensed under the Apache License 2.0.
See LICENSE.txt for more details.
 
Authors:
Hideaki Shimazaki (shimazaki.hideaki.8x@kyoto-u.jp) shimazaki on Github
Lee A.D. Cooper (cooperle@gmail.com) cooperlab on GitHub
Subhasis Ray (ray.subhasis@gmail.com)
 
Three methods are implemented in this package:
1. sshist - can be used to determine the optimal number of histogram bins for independent 
identically distributed samples from an underlying one-dimensional distribution. The
principal here is to minimize the L2 norm of the difference between the histogram and the
underlying distribution.

2. sskernel - implements kernel density estimation with a single globally-optimized 
bandwidth.

3. ssvkernel - implements kernel density estimation with a locally variable bandwidth.
 
Dependencies: These functions in this package depend on NumPy for various operations 
including fast-fourier transforms and histogram generation.