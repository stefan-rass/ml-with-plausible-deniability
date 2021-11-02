# Supervised Machine Learning with Plausible Deniability
Supplementary material for research work about machine learning with plausible deniability. The reference is:

S. Rass, S. König, J. Wachter, M. Egger, M. Hobisch: *Supervised Machine Learning with Plausible Deniability*, Elsevier Computers & Security (in press)
preprint available from arXiv:2106.04267v1 [cs.LG], [Online] https://arxiv.org/abs/2106.04267

The code runs under GNU Octave (version 6.2.0), obtainable from https://www.octave.org)
and configured for "x86_64-w64-mingw32"

Package dependencies:
optim, Version 1.6.0 (available from https://octave.sourceforge.io/optim/)

The code consists of the following files, along with explanations of their purpose
File									    Purpose
- experiments_local_opt.m   the main script; to load and run "as is" on the Octave prompt
- norm_v.m								  the norm constructed on the vector subspace V
- norm_w1.m								  the norm constructed on the vector subspace W_1
- seminorm_b.m							the seminorm b(x)
- crafted_norm.m						the final norm as constructed in the proof of Theorem 2
- execution_snapshot.txt	  a snapshot from an execution, added for convenience of the 
										      reader, who may not want to download and install Octave. This
										      is the data that was included in the paper
										
Remark on reproducibility of randomness when executing the code, one would have to seed the random
generator by extracting its current state
v = rand ("state")
saving this state persistently, and initializing the random generator with the (same) state before
a new repetition of the code: 
rand ("state", v)
We did not include this bit in the code, to let the code run on fresh instances of ML models each time.

Remark on further examples (related to logistic regression and neural networks): the code for
these examples is based on and generalizes the above program, yet more complex to run and use (since the output
is, due to the models therein, much more verbose and less "pretty-printed"). It will be uploaded soon.

For questions, please send an email to stefan.rass@jku.at. I am happy to answer and help!

Best wishes,
Stefan
