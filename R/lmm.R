#' Fitting univariate and multiviarate linear mixed models via data transforming augmentation
#'
#'  The function \code{lmm} fits univariate and multivariate linear mixed models
#'  (also called two-level Gaussian hierarchical models) whose first-level hierarchy is about
#'  a distribution of observed data and second-level hierarchy is about a prior distribution of random effects.
#'  For each group \eqn{i}, let \eqn{y_{i}} be an unbiased estimate of random effect \eqn{\theta_{i}},
#'  and \eqn{V_{i}} be a known measurement error variance.
#'
#'  The linear mixed model of interest is specified as follows:
#'  \deqn{[y_{i} \mid \theta_{i}] \sim N(\theta_{i}, V_{i})}
#'  \deqn{[\theta_{i} \mid \mu_{0i}, A) \sim N(\mu_{0i}, A)}
#'  \deqn{\mu_{0i} = x_{i}'\beta} independently for \eqn{i = 1, \ldots, k}, where \emph{k} is the number of groups (units) and dimension of each element is appropriately adjusted in a multivariate case.
#'
#'  The function \code{lmm} returns either maximum likelihood estimates or posterior samples of hyper-parameters \eqn{\beta} and \eqn{A}, the unknown parameters of the prior distributions for random effects. We adopt a Gibbs sampler based on standard family distributions.
#'
#'
#' @param y Response variable. In a univariate case, it is a vector of length \eqn{k} for the observed data. In a multivariate case, it is a (\eqn{k} by \eqn{p}) matrix, where \eqn{k} is the number of observations and \eqn{p} denotes the dimensionality.
#' @param v Known measurement error variance. In a univariate case, it is a vector of length \eqn{k}. In a multivariate case, it is a (\eqn{p}, \eqn{p}, \eqn{k}) array of known measurement error covariance matrices, i.e., each of \eqn{k} array components is a (\eqn{p} by \eqn{p}) covariance matrix.
#' @param x (Optional) Covariate information. If there is one covariate for each object, e.g., weight, it is a  vector of length \eqn{k} for the weight. If there are two covariates for each object, e.g., weight and height, it is a (\eqn{k} by 2) matrix, where each column contains each covariate variable. Default is no covariate (\code{x = 1}).
#' @param n.burn Number of warming-up iterations for a Markov chain Monte Carlo method. It must be specified for \code{method = "mcmc"}
#' @param n.sample Number of iterations (size of a posterior sample) for a Markov chain Monte Carlo method. It must be specified for \code{method = "mcmc"}
#' @param tol Tolerance that determines the stopping rule of the EM algorithm. The EM algorithm iterates until the change in log-likelihood function is within the tolerance. Default is 1e-10.
#' @param method \code{"em"} will return maximum likelihood estimates of the unknown hyper-parameters and \code{"mcmc"} returns posterior samples of the unknown parameters.
#' @param dta a logical; Data transforming augmentation is used if \code{dta = TRUE}, and typical data augmentation is used if \code{dta = FALSE}.
#' @param print.time a logical; \code{TRUE} to display two time stamps for initiation and termination, \code{FALSE} otherwise.
#'
#' @return The outcome of \code{lmm} is composed of:
#' \describe{
#' \item{A}{If \code{method} is "mcmc". It contains the posterior sample of \eqn{A}.}
#' \item{beta}{If \code{method} is "mcmc". It contains the posterior sample of \eqn{\beta}.}
#' \item{A.mle}{If \code{method} is "em". It contains the maximum likelihood estimate of \eqn{A}.}
#' \item{beta.mle}{If \code{method} is "em". It contains the maximum likelihood estimate of \eqn{beta}.}
#' \item{A.trace}{If \code{method} is "em". It contains the update history of \eqn{A} at each iteration.}
#' \item{beta.trace}{If \code{method} is "em". It contains the update history of \eqn{beta} at each iteration.}
#' \item{n.iter}{If \code{method} is "em". It contains the number of EM iterations.}
#' }
#'
#' @details
#' The function \code{lmm} produces maximum likelihood estimates,
#' their update histories of EM iterations, and the number of EM iterations if \code{method} is \code{"em"}.
#' It produces the posterior samples if \code{method} is \code{"bayes"}.
#'
#' @examples
#' ### Univariate linear mixed model
#'
#' # response variable for 10 objects
#' y <- c(5.42, -1.91, 2.82, -0.14, -1.83, 3.44, 6.18, -1.20, 2.68, 1.12)
#' # corresponding measurement error standard deviations
#' se <- c(1.05, 1.15, 1.22, 1.45, 1.30, 1.29, 1.31, 1.10, 1.23, 1.11)
#' # one covariate information for 10 objects
#' x <- c(2, 3, 0, 2, 3, 0, 1, 1, 0, 0)
#'
#' ## Fitting without covariate information
#' # (DTA) maximum likelihood estimates of A and beta via an EM algorithm
#' res <- lmm(y = y, v = se^2, method = "em", dta = TRUE)
#' # (DTA) posterior samples of A and beta via an MCMC method
#' res <- lmm(y = y, v = se^2, n.burn = 1e1, n.sample = 1e1,
#'            method = "mcmc", dta = TRUE)
#' # (DA) maximum likelihood estimates of A and beta via an EM algorithm
#' res <- lmm(y = y, v = se^2, method = "em", dta = FALSE)
#' # (DA) posterior samples of A and beta via an MCMC method
#' res <- lmm(y = y, v = se^2, n.burn = 1e1, n.sample = 1e1,
#'            method = "mcmc", dta = FALSE)
#'
#' ## Fitting with the covariate information
#' # (DTA) maximum likelihood estimates of A and beta via an EM algorithm
#' res <- lmm(y = y, v = se^2, x = x, method = "em", dta = TRUE)
#' # (DTA) posterior samples of A and beta via an MCMC method
#' res <- lmm(y = y, v = se^2, x = x, n.burn = 1e1, n.sample = 1e1,
#'            method = "mcmc", dta = TRUE)
#' # (DA) maximum likelihood estimates of A and beta via an EM algorithm
#' res <- lmm(y = y, v = se^2, x = x, method = "em", dta = FALSE)
#' # (DA) posterior samples of A and beta via an MCMC method
#' res <- lmm(y = y, v = se^2, x = x, n.burn = 1e1, n.sample = 1e1,
#'            method = "mcmc", dta = FALSE)
#'
#'
#' ### Multivariate linear mixed model
#'
#' # (arbitrary) 10 hospital profiling data (two response variables)
#' y1 <- c(10.19, 11.53, 16.28, 12.32, 12.84, 11.85, 14.81, 13.24, 14.43, 9.35)
#' y2 <- c(12.06, 14.97, 11.50, 17.88, 19.21, 14.69, 13.96, 11.07, 12.71, 9.63)
#' y <- cbind(y1, y2)
#'
#' # making measurement error covariance matrices for 10 hospitals
#' n <- c(24, 34, 38, 42, 49, 50, 79, 84, 96, 102) # number of patients
#' v0 <- matrix(c(186.87, 120.43, 120.43, 250.60), nrow = 2) # common cov matrix
#' temp <- sapply(1 : length(n), function(j) { v0 / n[j] })
#' v <- array(temp, dim = c(2, 2, length(n)))
#'
#' # covariate information (severity measure)
#' severity <- c(0.45, 0.67, 0.46, 0.56, 0.86, 0.24, 0.34, 0.58, 0.35, 0.17)
#'
#' ## Fitting without covariate information
#' # (DTA) maximum likelihood estimates of A and beta via an EM algorithm
#' res <- lmm(y = y, v = v, method = "em", dta = TRUE)
#' # (DTA) posterior samples of A and beta via an MCMC method
#' res <- lmm(y = y, v = v, n.burn = 1e1, n.sample = 1e1,
#'            method = "mcmc", dta = TRUE)
#' # (DA) maximum likelihood estimates of A and beta via an EM algorithm
#' res <- lmm(y = y, v = v, method = "em", dta = FALSE)
#' # (DA) posterior samples of A and beta via an MCMC method
#' res <- lmm(y = y, v = v, n.burn = 1e1, n.sample = 1e1,
#'           method = "mcmc", dta = FALSE)
#'
#' ## Fitting with the covariate information
#' # (DTA) maximum likelihood estimates of A and beta via an EM algorithm
#' res <- lmm(y = y, v = v, x = severity, method = "em", dta = TRUE)
#' # (DTA) posterior samples of A and beta via an MCMC method
#' res <- lmm(y = y, v = v, x = severity, n.burn = 1e1, n.sample = 1e1,
#'            method = "mcmc", dta = TRUE)
#' # (DA) maximum likelihood estimates of A and beta via an EM algorithm
#' res <- lmm(y = y, v = v, x = severity, method = "em", dta = FALSE)
#' # (DA) posterior samples of A and beta via an MCMC method
#' res <- lmm(y = y, v = v, x = severity, n.burn = 1e1, n.sample = 1e1,
#'            method = "mcmc", dta = FALSE)
#
#'@references
#' \insertRef{tak_data_2019}{Rdta}
#'
#' @author Hyungsuk Tak (maintainer), Kisung You, Sujit K. Ghosh, Bingyue Su, and Joseph Kelly
#' @export
lmm <- function(y, v, x = 1, n.burn, n.sample, tol = 1e-10, method = "em", dta = TRUE, print.time=FALSE) {

  # y : A (k by p) matrix for observations.
  #     The i-th row indicates the p-variate observations
  #     for the i-th individual.

  # v : A p by p by k array for the known measurement covariance matrices.
  #     The p by p measurement covariance matrix for the i-th individual
  #     is saved in the i-th array (i = 1, 2, ..., k).
  #     If p = 1, v can be entered as a vector of measurement variances of length k.

  # x : A rectangular matrix.
  #     The i-th row contains covariate information (measurements) of the i-th individual.
  #     Later, a vector of ones will be automatically added as the first column even when
  #     there is no covariate information (m = 1)
  #     When there is no covariate, a default is x = matrix(1, nrow = k, ncol = 1).

  # n.burn : The number of burn-in iterations to be discarded.

  # n.sample : The number of MCMC sample to be obtained.

  # tol : tolerance for the EM iterations. If the change of the log-likelihood in each iteration,
  #       i.e., abs((log-likelihood in the previous iteration)
  #                 - (log-likelihood in the current iteration)) /
  #             (log-likelihood in the previous iteration),
  #       is smaller than "tol", then the iteration stops.
  #       The default is "1e-10".

  # method : If "em", the output contains the maximum likelihood estimates of unknown parameters, A and beta.
  #          If "mcmc", the output contains the posterior sample of unknown model parameters, A and beta.
  #          The default is "em".

  # dta : If TRUE, it fits the model via data transforming augmentation.
  #       If FALSE, it fits the model via typical data augmentation.

  if (is.matrix(y)) {
    temp <- dim(y)
    k <- temp[1]
    p <- temp[2]
    # k : The number of individual
    # p : The number of multivariate observations for each individual.
  } else {
    k <- length(y)
    p <- 1
    y <- as.matrix(y)
  }

  if (all(x == 1)) {
    x <- matrix(1, nrow = k, ncol = 1)
  } else {
    x <- cbind(rep(1, k), x)
  }

  # treat x, y as matrix even if p = 1
  # m : The number of regression coefficients
  m <- dim(x)[2]

  if (print.time){
    print(Sys.time())
  }

  if (method == "em") {
    # initial setting for the number of iteration and tolerance.
    n.iter <- 0
    tol.t <- 1

    # to keep the updating history
    beta.trace <- NULL
    A.trace <- NULL

    if (p == 1) {

      # univariate linear mixed model

      if (dta == TRUE) {
        v.min <- min(v)[1]
        w <- 1 - v.min / v
      }

      # initial values of parameters
      A.trace <- A.t <- v[sample(1 : k, 1)]
      beta.trace <- beta.t <- rnorm(m * p)

      # define the log-likelihood function to be used
      loglik <- function(beta, A, y, v, x) {
        sum(dnorm(y, mean = x %*% beta, sd = sqrt(A + v), log = TRUE))
      }

      # initial log-likelihood value
      loglik.t <- loglik(beta = beta.t, A = A.t, y = y, v = v, x = x)

      while (tol.t > tol) {
        n.iter <- n.iter + 1
        B <- v / (v + A.t)
        mu0 <- x %*% beta.t

        if (dta == TRUE) {
          mu.ast <- (1 - w * B) * y + w * B * mu0
          v.ast <- w * v.min + w^2 * (1 - B) * v
          A.t <- max(mean(v.ast + (mu.ast - mu0)^2) - v.min, 0)
        } else {
          mu.ast <- (1 - B) * y + B * mu0
          v.ast <- (1 - B) * v
          A.t <- mean(v.ast + (mu.ast - mu0)^2)
        }
        A.trace <- c(A.trace, A.t)

        if (m == 1) {
          beta.t <- mean(mu.ast)
          beta.trace <- c(beta.trace, beta.t)
        } else {
          beta.t <- chol2inv(chol(t(x) %*% x)) %*% t(x) %*% mu.ast
          beta.trace <- rbind(beta.trace, t(beta.t))
        }

        loglik.temp <- loglik(beta.t, A.t, y, v, x)
        tol.t <- abs(loglik.temp - loglik.t) / abs(loglik.t)
        loglik.t <- loglik.temp

      }

    } else {
      # multivariate linear mixed model (p >= 2)

      if (all(x == 1)) {
        x <- array(diag(p), c(p, p, k))
        m <- 1
      } else {
        m <- dim(x)[2]
        x <- array(sapply(1 : k, function(i) {
          kronecker(diag(p), t(x[i, ]))
        }), c(p, m * p, k))
      }

      if (dta == TRUE) {
        # the minimum covariance matrix
        v.eigen <- sapply(1 : k, function(i) { eigen(v[, , i])$values })
        v.min <- min(v.eigen) * diag(p)
        w <- array(sapply(1 : k, function(j) {
          diag(p) - v.min %*% solve(v[, , j])
        }), c(p, p, k))
      }

      # initial values
      A.t <- v[, , sample(1 : k, 1)]
      A.trace <- rbind(A.trace, as.numeric(A.t))
      beta.t <- rnorm(m * p)
      beta.trace <- rbind(beta.trace, t(beta.t))

      # define the log of the multivariate Normal density to be used just below.
      log.dmvnorm <- function(y, mean, sigma) {
        -p / 2 * log(2 * pi) -0.5 * log(det(sigma)) -
          0.5 * t(y - mean) %*% chol2inv(chol(sigma)) %*% (y - mean)
      }

      # define the log-likelihood function to be used below.
      loglik <- function(beta, A, y, v, x) {
        sum(sapply(1 : dim(y)[1], function(i) {
          log.dmvnorm(y[i, ], mean = x[, , i] %*% beta, sigma = A + v[, , i])
        }))
      }

      loglik.t <- loglik(beta = beta.t, A = A.t, y = y, v = v, x = x)

      while (tol.t > tol) {
        n.iter <- n.iter + 1
        B <- array(sapply(1 : k, function(u) {
          v[, , u] %*% chol2inv(chol(v[, , u] + A.t))
        }), c(p, p, k))
        mu0 <- t(sapply(1 : k, function(l) { x[, , l] %*% beta.t }))

        if (dta == TRUE) {
          z.mean <- t(sapply(1 : k, function(l) {
            (diag(p) - w[, , l] %*% B[, , l]) %*% y[l, ] +
              w[, , l] %*% B[, , l] %*% mu0[l, ]
          }))
          z.var <- array(sapply(1 : k, function(j) {
            w[, , j] %*% (diag(p) - B[, , j]) %*% v[, , j] %*% t(w[, , j]) +
              v.min %*% t(w[, , j])
          }), c(p, p, k))
        } else {
          z.mean <- t(sapply(1 : k, function(l) {
            (diag(p) - B[, , l]) %*% y[l, ] +
              B[, , l] %*% mu0[l, ]
          }))
          z.var <- array(sapply(1 : k, function(j) {
            (diag(p) - B[, , j]) %*% v[, , j]
          }), c(p, p, k))
        }

        res.temp <- z.mean - mu0
        A.t.rowmean <- rowMeans(sapply(1 : k, function(i) {
          res.temp[i, ] %*% t(res.temp[i, ]) + z.var[, , i]
        }))

        if (dta == TRUE) {
          A.temp <- array(A.t.rowmean, c(p, p)) - v.min
          if (det(A.temp) > 0) {
            A.t <- A.temp
            A.trace <- rbind(A.trace, as.numeric(A.temp))
          } else {
            A.t <- matrix(0, ncol = p, nrow = p)
            A.trace <- rbind(A.trace, as.numeric(A.t))
          }
        } else {
          A.t <- array(A.t.rowmean, c(p, p))
          A.trace <- rbind(A.trace, A.t.rowmean)
        }

        beta.t1 <- matrix(rowSums(sapply(1 : k, function(l) { t(x[, , l]) %*% x[, , l] })),
                          nrow = m * p, ncol = m * p)
        beta.t2 <- rowSums(sapply(1 : k, function(l) { t(x[, , l]) %*% z.mean[l, ] }))
        beta.t <- chol2inv(chol(beta.t1)) %*% beta.t2
        beta.trace <- rbind(beta.trace, t(beta.t))

        loglik.temp <- loglik(beta = beta.t, A = A.t, y = y, v = v, x = x)
        tol.t <- abs(loglik.temp - loglik.t) / abs(loglik.t)
        loglik.t <- loglik.temp
      }

    }

    out <- list(beta.mle = beta.t, A.mle = A.t, n.iter = n.iter,
                A.trace = A.trace, beta.trace = beta.trace)

  } else { # if method == "mcmc"
    n.total <- n.burn + n.sample

    if (p == 1) { # for a univariate linear mixed model

      A.out <- rep(NA, n.total)
      beta.out <- matrix(NA, nrow = n.total, ncol = m)
      # ncol = mp but p=1 for this univariate case

      if (dta == TRUE) {
        V.min <- min(v)
        W <- 1 - V.min / v
      }

      # initial values of parameters
      A.t <- v[sample(1 : k, 1)]
      beta.t <- rnorm(m * p)

      if (m == 1) { # No covariates (with only an intercept term)

        for (i in 1 : n.total) {

          # Equation (2.11) in Kelly (2014)
          B <- v / (v + A.t)

          if (dta == TRUE) {
            z.mean <- (1 - W * B) * y + W * B * beta.t
            z.var <- W^2 * v * (1 - B)  + V.min * W
            z.t <- rnorm(k, z.mean, sqrt(z.var))

            # Equation (2.7) in Kelly (2014)
            beta.hat <- mean(z.t)
            res.temp <- z.t - mean(z.t)
            inv.gamma.scale <- sum(res.temp^2) / 2

            A <- -1
            while (A <= 0) {
              A <- inv.gamma.scale / rgamma(1, shape = (k - 3) / 2) - V.min
            }
            A.out[i] <- A.t <- A

            # Equation (2.6) in Kelly (2014)
            beta.var <- (V.min + A.t) / k
            beta.out[i, ] <- beta.t <- rnorm(1, mean = beta.hat, sd = sqrt(beta.var))
          } else {
            theta.mean <- (1 - B) * y + B * beta.t
            theta.var <- (1 - B) * v
            theta.t <- rnorm(k, mean = theta.mean, sd = sqrt(theta.var))

            # Equation (2.7) in Kelly (2014)
            beta.hat <- mean(theta.t)
            A.scale <- sum((theta.t - beta.hat)^2) / 2
            A.t <- A.scale / rgamma(1, shape = (k - 3) / 2)
            A.out[i] <- A.t

            beta.var <- A.t / k
            beta.out[i] <- beta.t <- rnorm(1, mean = beta.hat, sd = sqrt(beta.var))
          } # end of "if (dta == TRUE)"
        } # end of "for (i in 1 : n.total)"

      } else { # m >= 2, but still p = 1, when there is covariate information

        for (i in 1 : n.total) {

          # Equation (2.11) in Kelly (2014)
          mu0 <- x %*% beta.t
          B <- v / (v + A.t)

          if (dta == TRUE) {
            z.mean <- (1 - W * B) * y + W * B * mu0
            z.var <- W^2 * v * (1 - B)  + V.min * W
            z.t <- rnorm(k, z.mean, sqrt(z.var))

            # Equation (2.7) in Kelly (2014)
            XX.inv <- chol2inv(chol(t(x) %*% x))
            beta.hat <- XX.inv %*% (t(x) %*% z.t)
            res.temp <- z.t - x %*% beta.hat
            inv.gamma.scale <- sum(res.temp^2) / 2
            A <- -1
            while (A <= 0) {
              A <- inv.gamma.scale / rgamma(1, shape = (k - m - 2) / 2) - V.min
            }
            A.t <- A.out[i] <- A

            # Equation (2.6) in Kelly (2014)
            beta.var <- as.numeric(V.min + A.t) * XX.inv
            beta.t <- beta.out[i, ] <- t(rmvnorm(1, mean = beta.hat, sigma = beta.var))
          } else {
            # Equation (2.11) in Kelly (2014)
            theta.mean <- (1 - B) * y + B * mu0
            theta.var <- (1 - B) * v
            theta.t <- rnorm(k, mean = theta.mean, sd = sqrt(theta.var))

            # Equation (2.7) in Kelly (2014)
            XX.inv <- chol2inv(chol(t(x) %*% x))
            beta.hat <- XX.inv %*% (t(x) %*% theta.t)
            res.temp <- theta.t - x %*% beta.hat
            inv.gamma.scale <- sum(res.temp^2) / 2
            A <- -1
            while (A <= 0) {
              A <- inv.gamma.scale / rgamma(1, shape = (k - m - 2) / 2)
            }
            A.t <- A.out[i] <- A

            # Equation (2.6) in Kelly (2014)
            beta.var <- as.numeric(A.t) * XX.inv
            beta.t <- beta.out[i, ] <- t(rmvnorm(1, mean = beta.hat, sigma = beta.var))
          } # end of "if (dta == TRUE) {} else {}"
        } # end of "for (i in 1 : n.total)"
      } # end of "if (m == 1) {} else {}"

      # output for p = 1
      out <- list(A = A.out[-c(1 : n.burn)], beta = beta.out[-c(1 : n.burn), ])

    } else { # if p >= 2 for a multivariate linear mixed model

      # initial values of parameters
      A.t <- v[, , sample(1 : k, 1)]
      beta.t <- rnorm(m * p)

      # make an array of covariate matrices X_i's (for i = 1, 2, ..., k)
      # using a kronecker product.
      # Each X_i is a (p by mp) matrix.
      X <- array(sapply(1 : k, function(i) {
        kronecker(diag(p), t(x[i, ]))
      }), c(p, m * p, k))

      A.out <- array(NA, dim = c(p, p, n.total))
      beta.out <- matrix(NA, nrow = n.total, ncol = p * m)

      if (dta == TRUE) {
        # the minimum covariance matrix
        v.eigen <- sapply(1 : k, function(i) { eigen(v[, , i])$values })
        v.min <- min(v.eigen) * diag(p)

        # weights
        w <- array(sapply(1 : k, function(i) {
          diag(p) - v.min %*% solve(v[, , i])
        }), c(p, p, k))
      }

      # defining a function for sampling A
      mean.beta.given.z <- function(Y, X) {
        var.beta <- chol2inv(chol(matrix(rowSums(sapply(1 : k, function(l) {
          t(X[, , l]) %*% X[, , l]
        })), nrow = m * p, ncol = m * p)))
        mean.beta <- var.beta %*% rowSums(sapply(1 : k, function(l) {
          t(X[, , l]) %*% Y[, l] }))
        mean.beta
      }

      if (dta == TRUE) {
        # defining a function for sampling beta
        mean.var.beta.given.z.A.dta <- function(Y, V, A, X) {
          VA.inv <- chol2inv(chol(V + A))
          var.beta <- chol2inv(chol(matrix(rowSums(sapply(1 : k, function(l) {
            t(X[, , l]) %*% VA.inv %*% X[, , l]
          })), nrow = m * p, ncol = m * p)))
          mean.beta <- var.beta %*% rowSums(sapply(1 : k, function(l) {
            t(X[, , l]) %*% VA.inv %*% Y[, l]
          }))
          list(mean = mean.beta, var = var.beta)
        }
      } else {
        # defining a function for sampling beta
        mean.var.beta.given.z.A.da <- function(Y, A, X) {
          A.inv <- chol2inv(chol(A))
          var.beta <- chol2inv(chol(matrix(rowSums(sapply(1 : k, function(l) {
            t(X[, , l]) %*% A.inv %*% X[, , l]
          })), nrow = m * p, ncol = m * p)))
          mean.beta <- var.beta %*% rowSums(sapply(1 : k, function(l) {
            t(X[, , l]) %*% A.inv %*% Y[, l]
          }))
          list(mean = mean.beta, var = var.beta)
        }
      }

      for (i in 1 : n.total) {

        # computing shrinkage B
        B <- array(sapply(1 : k, function(u) {
          v[, , u] %*% chol2inv(chol(v[, , u] + A.t))
        }), c(p, p, k))

        if (dta == TRUE) {
          # computing the mean of the Gaussian distribution for complete data
          z.mean <- as.matrix(sapply(1 : k, function(l) {
            (diag(p) - w[, , l] %*% B[, , l]) %*% y[l, ] +
              w[, , l] %*% B[, , l] %*% X[, , l] %*% beta.t
          }))

          # computing the covariance of the Gaussian distribution for complete data
          z.var <- array(sapply(1 : k, function(j) {
            w[, , j] %*% (diag(p) - B[, , j]) %*% v[, , j] %*% t(w[, , j]) +
              v.min %*% t(w[, , j])
          }), c(p, p, k))

          # update the complete data
          z.t <- sapply(1 : k, function(u) {
            t(rmvnorm(1, mean = z.mean[, u], sigma = z.var[, , u]))
          })

          # updating A given the complete data
          # compute beta.hat given the complete data
          beta.hat <- mean.beta.given.z(Y = z.t, X = X)
          S <- matrix(rowSums(sapply(1 : k, function(l) {
            res.temp <- z.t[, l] - X[, , l] %*% beta.hat
            res.temp %*% t(res.temp)
          })), nrow = p, ncol = p)

          # repeatedly sample A + V.min from inverse-Wishart
          # and subtract V.min until the resulting matrix is non-negative definite
          dt <- -1
          while (dt < 0) {
            A.t <- riwish(v = k - m - p - 1, S = S) - v.min
            dt <- det(A.t)
          }
          A.out[, , i] <- A.t

          # updating beta given A and the complete data
          beta.temp <- mean.var.beta.given.z.A.dta(Y = z.t, V = v.min, A = A.t, X = X)
          beta.t <- beta.out[i, ] <- t(rmvnorm(1, mean = beta.temp$mean,
                                               sigma = beta.temp$var))
        } else {
          # mean of the Gaussian distribution for complete data
          z.mean <- as.matrix(sapply(1 : k, function(l) {
            (diag(p) - B[, , l]) %*% y[l, ] +
              B[, , l] %*% X[, , l] %*% beta.t
          }))

          # covariance of the Gaussian distribution for complete data
          z.var <- array(sapply(1 : k, function(j) {
            (diag(p) - B[, , j]) %*% v[, , j]
          }), c(p, p, k))

          # update the complete data
          z.t <- sapply(1 : k, function(u) {
            z.mean[, u] + chol(z.var[, , u]) %*% rnorm(p)
          })

          # compute beta.hat given the complete data.
          beta.hat <- mean.beta.given.z(Y = z.t, X = X)
          S <- matrix(rowSums(sapply(1 : k, function(l) {
            res.temp <- z.t[, l] - X[, , l] %*% beta.hat
            res.temp %*% t(res.temp)
          })), nrow = p, ncol = p)

          # update A
          A.t <- riwish(v = k - m - p - 1, S = S)
          A.out[, , i] <- A.t

          beta.temp <- mean.var.beta.given.z.A.da(Y = z.t, A = A.t, X = X)
          beta.t <- beta.out[i, ] <- t(rmvnorm(1, mean = beta.temp$mean,
                                               sigma = beta.temp$var))
        } # end of "if (dta == TRUE) {} else {}"
      } # end of "for (i in 1 : n.total) {}"

      A <- A.out[, , -c(1 : n.burn)]
      beta <- beta.out[-c(1 : n.burn), ]

      # output
      out <- list(A = A, beta = beta)
    } # end of "if (p == 1) {} else {}"
  } # end of "if (method == "em") {} else {}"

  if (print.time){
    print(Sys.time())
  }

  return(out)

}
