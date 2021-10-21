function v = crafted_norm(x, B, w1) 
  
  % this is the norm from theorem 1: b(x) + (norm on V + norm on W1) 
  v = seminorm_b(x,B) + norm_v(x,B) + norm_w1(x,B, w1);
  
endfunction