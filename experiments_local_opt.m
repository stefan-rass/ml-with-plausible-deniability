% experimental test for local optimality

clear all   % avoid side-effects upon several executions
pkg load optim  % for nonlinear optimization 

% test with reasonably small dimensions; scalability is indeed an issue, not the 
% least so because the numeric optimization becomes much harder. Our goal is using
% an "as-is" optimizer as ships with the Octave system; more sophisticated optimizers
% will be considered in follow up work. 
% This is ONLY A PROOF OF CONCEPT (to show that the idea works)
n = 10   % number of training data records
m = 5   % dimension of x (training data) 
d = m+1 % d=m+1 in our case, since we have the intercept as an additional paramter

x_training = randi(8, n, m)  % random (assumed to be unknown) training data
p_true = 6*rand(d,1) - 3   % let the model be simply a random one here...
% let the random data be exponentially distributed. 
y_training = p_true(1) + x_training * p_true(2:d) + (-1/5*log(rand(n,1)))

% train the ML model, and let it be given as the parameter vector p
[p, objf, cvg, outp] = nonlin_min(@(beta) (norm(beta(1) + x_training*beta(2:d) - y_training)), zeros(d,1));
% the "exact" result is also directly obtainable via the pseudoinverse: the call 
% pinv([ones(n,1) x_training])*y_training
% should give the same result as "p" from the nonlinear optimization (but more "precisely")
if (cvg < 0) 
  disp('1st optimization failed')
endif

x_decoy = randi(8,n,m)  % this shall be the data that we seek to make plausible
% note that we now pick the response also at random!
%y_decoy = (-1/2*log(rand(n,1)))   % sample from the same distribution (plausibility) but with a different parameter
y_decoy = randi(8,n,1)  % alternatively, we can also take a different distribution (works equally well)

% get the error vector
e = y_decoy - (p(1) + x_decoy * p(2:d))   % evaluate the ML model

% now, construct the norm for the denial
% check the rank condition, Jacobian is closed form expressible from the training data
J = [ones(n,1), x_training]
if (rank([J, e]) == rank(J)) 
  disp('rank condition failed; re-run to pick new training data!')
else
  % the rank condition was satisfied
  disp('rank condition OK; constructing the norm')
  B = (null(e'))';
  
  % constructing a linearly independent subspace of dimension 1
  w1 = e;
  w1(1) = w1(1) + 1;  % small distortion of e should be enough
  %w1 = 1/2*e + 1/2*(B(1,:))';  % alternative choice, guaranteed to be linearly independent, but not necessarily numerically very stable
  while(rank([w1 e]) < 2 || rank(B) >= rank([B;w1']))
    w1 = rand(size(e))  % retry with a random vector (should succeed with probability 1)
  endwhile
  w1 = w1 / sum(w1);  % scale to unit length

  disp('re-fitting')
  % re-fit with the new norm functional
  p0 = p + 0.1*rand;  % start within a proximity; *local* optimality only (as our model is linear, we could - in theory - have used any starting point, though)
  [p2, objf, cvg, outp] = nonlin_min(@(beta) (crafted_norm(beta(1) + x_decoy*beta(2:d) - y_decoy, B, w1)), p0);
  if (cvg < 0) 
    disp('2nd optimization failed')
  endif

  % the newly found result should be (roughly) equal to the original parameter vector 
  % (up to perhaps numeric inaccuracies, but approximately recover p)
  disp('comparing the results: [original | newly fitted]')
  disp([p p2])
endif
