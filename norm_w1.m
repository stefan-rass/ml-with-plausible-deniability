function v = norm_w1(x,B,w1) 
  
  % projection on the 1-dim. subspace spanned by w1
  alpha = seminorm_b(w1, B);  % this value is constant, and could (in a more efficient implementation) be supplied externally or globally
  lambda = (w1' * x);
  y1 = lambda * w1;

  v = 0.5 * alpha * norm(y1, 1);  
  
endfunction