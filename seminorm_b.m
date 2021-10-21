function v = seminorm_b(x, B) 
  v = sqrt(x'*B'*B*x);  % 2-norm induced by the matrix B
endfunction
