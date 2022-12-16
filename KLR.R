l1_dist <- function(u, w){
  sum(abs(u-w))
}

#
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

KLR = function(
    y, #binary response
    x, #data
    kernel = c("polynomial", "RBF", "laplace", "sigmoid")[1], #types of kernel
    lambda = 0.01, #hyperparameters
    sigma2 = 1,
    c = 1,
    d = 3,
    tol = 1e-6, #convergence criteria
    max_iter = 1e5
){
  # Error check
  if(is.vector(y)) y = matrix(y, length(y), 1)
  if(lambda<1.0e-16) stop("lambda should be positive")
  if(sigma2<1.0e-16) stop("sigma2 should be positive")
  if(tol<1.0e-16) stop("tol should be positive")
  if(d < 1) stop("d should be larger than or equal to 1")
  
  # Preparation
  n = nrow(x)
  p = ncol(x)
  
  KERNELS = c("polynomial", "RBF", "laplace", "sigmoid")
  
  # Compute kernels
  if(kernel == "polynomial"){
    D = tcrossprod(x)
    K = (D + c)^d
  }else if(kernel == "RBF"){
    D = as.matrix(stats::dist(x))
    K = exp(-D^2 / (2*sigma2))
  }else if (kernel == "laplace") {
    D = as.matrix(stats::dist(x, method = "manhattan"))
    K = exp(-D / sqrt(sigma2))
  }else if (kernel == "sigmoid") {
    xs = scale(x, scale=F)
    D = tcrossprod(xs)
    K = tanh(D + c)
  }
  else{
    stop(paste("only kernels", KERNELS, "are implemented"))
  }
  
  # obj function
  fn = function(alpha){
    lin_pred = K %*% alpha
    penalty = lambda/2 * crossprod(alpha, K %*% alpha)
    loss = mean(-y * lin_pred + log(1 + exp(lin_pred)))
    return((loss + penalty))
  }
  
  
  # gradient wrt alpha
  gr = function(alpha){
    lin_pred = K %*% alpha
    prob = 1 / (1 + exp(-lin_pred))
    g_alpha = -crossprod(K, (y - prob - lambda * alpha))
    return(g_alpha)
  }
  
  # fit using gradient descent
  alpha = rep(0, n)
  results = stats::optim(par = alpha, fn = fn, gr = gr, method = "BFGS", control = list(maxit = max_iter, abstol = tol))
  
  # return
  out = list(x=x, alpha=results$par, counts = results$counts, NLL = results$value, 
             kernel=kernel, lambda = lambda, sigma2 = sigma2, c = c, d = d)
  return(out)
}

predict.KLR = function(object, newx=object$x, ...){
  m = nrow(newx)
  n = nrow(object$x)
  p = ncol(object$x)
  
  #Construct Kernels
  if (object$kernel == "polynomial"){
    D = tcrossprod(object$x, newx)
    K = (D + object$c)^object$d
  }else if(object$kernel == "RBF"){
    D = matrix(pdist::pdist(object$x, newx)@dist, n, m, T)
    K = exp(-D^2 / (2 * object$sigma2))
  }else if(object$kernel == "laplace") {
    D = l1_norm(newx, object$x)
    K = exp(-D / sqrt(object$sigma2))
  }else if(object$kernel == "sigmoid") {
    xs = scale(object$x, scale=F)
    newx = (newx - matrix(attr(xs, 'scaled:center'), m, p, T))
    D = tcrossprod(xs, newx)
    K = tanh(D + object$c)
  }
  
  #Compute Prob
  f = crossprod(K, object$alpha)
  p = 1 / (1 + exp(-f))
  
  return(p)
}