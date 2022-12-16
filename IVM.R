# Helper function to compute L1 norm of two vectors
l1_dist <- function(u, w){
  sum(abs(u-w))
}

# Helper function to compute L1 norm of two matrices/vector
l1_norm <- function(x1, x2) {
  if (is.vector(x1) & is.vector(x2)) {
    l1_dist(x1, x2)
  }else if (is.vector(x1) | is.vector(x2)) {
    apply(x1, 1, FUN = l1_dist, w = x2)
  }else{
    f <- function(v) {
      apply(x2, 1, FUN = l1_dist, w = v)
    }
    apply(x1, 1, FUN = f)
  }
}

IVM <- function(
    y, #Binary response
    x, #Data
    kernel = c("polynomial", "RBF", "laplace", "sigmoid")[1], #kernels
    lambda = 0.01, #hyperparameters
    sigma2 = 1,
    c = 1,
    d = 3,
    tol = 1e-6, #Convergence Criteria
    max_iter = 1e5
) {
  
  n = nrow(x)
  p = ncol(x)
  
  # Error check
  if(is.vector(y)) y = matrix(y, length(y), 1)
  if(lambda<1.0e-16) stop("lambda should be positive")
  if(sigma2<1.0e-16) stop("sigma2 should be positive")
  if(tol<1.0e-16) stop("tol should be positive")
  if(d < 1) stop("d should be larger than or equal to 1")
  
  KERNELS = c("polynomial", "RBF", "laplace", "sigmoid")
  
  # Compute Kernels for the First Iteration
  # S is empty, R has all obs
  if(kernel == "polynomial"){
    D_a = apply(x, 1, function(x_n) {x %*% x_n}) 
    K_a = (D_a + c)^d # each col is K_al; nxn
    D_q = apply(x, 1, crossprod)
    K_q = matrix((D_q + c)^d, nrow = 1) # each col (scaler) is K_q; nx1
  }else if(kernel == "RBF"){
    K_a = apply(x, 1, function(x_n) {
      v = pdist::pdist(x, x_n)@dist
      return(exp(-v^2/ (2*sigma2)))
    })
    K_q = matrix(rep(0, n), nrow = 1)
  }else if (kernel == "laplace") {
    sigma = sqrt(sigma2)
    K_a = apply(x, 1, function(x_n) {
      v = l1_norm(x, x_n)
      return(exp(-v / sigma))
    })
    K_q = matrix(rep(0, n), nrow = 1)
  }else if (kernel == "sigmoid") {
    xs = scale(x, scale=F)
    D_a = apply(xs, 1, function(x_n) {xs %*% x_n}) 
    K_a = tanh(D_a + c) # each col is K_al; nxn
    D_q = apply(xs, 1, crossprod)
    K_q = matrix(tanh(D_q + c), nrow = 1) # each col (scaler) is K_q; nx1
  }
  else{
    stop(paste("only kernels", KERNELS, "are implemented"))
  }
  
  Hl = rep(NA, n)
  as = rep(NA, n)
  for (i in 1:n) {
    K_al = K_a[, i]
    K_ql = K_q[, i]
    
    # objective function
    fn = function(alpha){
      lin_pred = K_al * alpha
      penalty = lambda/2 * alpha * K_ql * alpha
      loss = mean(-y * lin_pred + log(1 + exp(lin_pred)))
      return((loss + penalty))
    }
    
    # gradient wrt alpha
    gr = function(alpha){
      lin_pred = K_al * alpha
      prob = 1 / (1 + exp(-lin_pred))
      g = -crossprod(K_al, (y - prob)) - K_ql * lambda * alpha
      return(g)
    }
    
    results = stats::optim(par = 0, fn = fn, gr = gr, method = "BFGS", 
                           control = list(maxit = max_iter, abstol = tol))
    
    Hl[i] = results$value
    as[i] = results$par
  }
  
  H_min = min(Hl)
  S_ind = which.min(Hl)
  R_ind = (1:n)[-S_ind]
  S = x[S_ind, ]
  R = x[-S_ind, ]
  K_a_min = K_a[, S_ind] #should be nx1
  K_q_min = K_q[, S_ind] #should be 1x1
  par = as[S_ind]
  
  #find S until converge
  convergence = 0
  while (convergence != 1 & length(R_ind) > 1) {
    if(kernel == "polynomial"){
      D_a = apply(R, 1, function(x_n) {x %*% x_n})
      K_a = (D_a + c)^d # every col is K_al; nxn
      
      D_q = apply(R, 1, function(x_n) {S %*% x_n})
      D_q_d = apply(R, 1, crossprod)
      K_q = (D_q + c)^d #each col (scaler) is K_q; nx1
      K_q_d = (D_q_d + c)^d #rx1
    }else if(kernel == "RBF"){
      K_a = apply(R, 1, function(x_n) {
        v = pdist::pdist(x, x_n)@dist
        return(exp(-v^2/ (2*sigma2)))
      })
      K_q = apply(R, 1, function(x_n) {
        v = pdist::pdist(S, x_n)@dist
        return(exp(-v^2/ (2*sigma2)))
      })
      K_q_d = matrix(0, nrow = 1, ncol = length(R_ind))
    }else if (kernel == "laplace") {
      K_a = apply(R, 1, function(x_n) {
        v = l1_norm(x, x_n)
        return(exp(-v / sigma))
      })
      K_q = apply(R, 1, function(x_n) {
        v = l1_norm(S, x_n)
        return(exp(-v / sigma)) 
      })
      K_q_d = matrix(0, nrow = 1, ncol = length(R_ind))
    }else if (kernel == "sigmoid") {
      D_a = apply(R, 1, function(x_n) {xs %*% x_n})
      K_a = tanh(D_a + c) # every col is K_al; nxn
      
      D_q = apply(R, 1, function(x_n) {S %*% x_n})
      D_q_d = apply(R, 1, crossprod)
      K_q = tanh(D_q + c) #each col (scaler) is K_q; nx1
      K_q_d = tanh(D_q_d + c) #rx1
    }
    
    # Augment the previously calculated kernels to save time
    q = length(S_ind)
    Hl = rep(NA, n-q)
    as = matrix(NA, nrow = n-q, ncol = q+1) #n-1 x's to be selected, with length(a) = a+1
    for (i in 1:(n-q)) {
      K_al = cbind(K_a_min, K_a[, i])
      if (length(S_ind) == 1) {
        K_q = matrix(K_q, nrow = 1)
      }
      K_ql = cbind(K_q_min, K_q[, i])
      K_q_row = c(K_q[, i], K_q_d[i])
      K_ql = rbind(K_ql, K_q_row)
      
      # objective function
      fn = function(alpha){
        lin_pred = K_al %*% alpha
        penalty = lambda/2 * crossprod(alpha, K_ql %*% alpha)
        loss = mean(-y * lin_pred + log(1 + exp(lin_pred)))
        return((loss + penalty))
      }
      
      # gradient
      gr = function(alpha){
        lin_pred = K_al %*% alpha
        prob = 1 / (1 + exp(-lin_pred))
        g = -crossprod(K_al, (y - prob)) - crossprod(K_ql, lambda*alpha)
        return(g)
      }
      
      a = rep(0, q+1) #q+1
      results = stats::optim(par = a, fn = fn, gr = gr, method = "BFGS", 
                             control = list(maxit = max_iter, abstol = tol))
      
      Hl[i] = results$value
      as[i, ] = results$par
    }
    
    #Choose the optim xl that minimize Hl
    H_new_min = min(Hl)
    min_ind = which.min(Hl)
    K_a_min = cbind(K_a_min, K_a[, min_ind]) #should be nx(q+1)
    K_q_min = cbind(K_q_min, K_q[, min_ind])
    K_q_min_row = c(K_q[, min_ind], K_q_d[min_ind])
    K_q_min = rbind(K_q_min, K_q_row) #should be (q+1)x(q+1)
    
    min_x_ind = R_ind[min_ind]
    S_ind = c(S_ind, min_x_ind)
    R_ind = (1:n)[-S_ind]
    S = x[S_ind, ]
    R = x[R_ind, ]
    par = as[min_ind, ]
    
    if (abs(H_new_min - H_min)/abs(H_new_min) < 0.001) {
      convergence = 1
    }
    H_min = H_new_min
  }
  
  #if doesn't converge, use KLR (i.e. using all obs)
  if (!convergence) {
    S = x
    S_ind = 1:n
    results = KLR(y, x, kernel, lambda, sigma2, c, d = 3, tol, max_iter)
    par = results$par
  }
  
  return(list(x = x, S = S, alpha = par, S_ind = S_ind, kernel=kernel, 
              lambda = lambda, sigma2 = sigma2, c = c, d = d, convergence = convergence))
}

predict.IVM = function(object, newx=object$x, ...){
  # construct kernel
  m = nrow(newx)
  n = nrow(object$S)
  p = ncol(object$S)
  
  if (object$kernel == "polynomial"){
    D = tcrossprod(object$S, newx)
    K = (D + object$c)^object$d
  }else if(object$kernel == "RBF"){
    D = matrix(pdist::pdist(object$S, newx)@dist, n, m, T)
    K = exp(-D^2 / (2 * object$sigma2))
  }else if(object$kernel == "laplace") {
    D = l1_norm(object$S, newx)
    K = t(exp(-D / sqrt(object$sigma2)))
  }else if(object$kernel == "sigmoid") {
    xs = scale(object$S, scale=F)
    newx = (newx - matrix(attr(xs, 'scaled:center'), m, p, T))
    D = tcrossprod(xs, newx)
    K = tanh(D + object$c)
  }
  
  # Compute Prob
  f = crossprod(K, object$alpha)
  p = 1 / (1 + exp(-f))
  
  return(p)
}

