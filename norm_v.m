function v = norm_v(x, B) 
  
  y = B'*B*x;  % projection on the row-space of B
  % note that since B*B'=I by construction, we would have b(y) = b(x), so
  % the projection could even be spared at all :-)
 
  % use the semi-norm on the so-projected vector
  v = 0.5 * seminorm_b(y, B);
  
endfunction
