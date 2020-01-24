#' Data Transforming Augmentation for Linear Mixed Models
#'
#' The R package \pkg{Rdta} provides a toolbox to fit univariate and multivariate linear mixed models via data transforming augmentation. Users can also fit these models via typical data augmentation for a comparison. It returns either maximum likelihood estimates of unknown model parameters via an EM algorithm or posterior samples via a Markov chain Monte Carlo method.
#'
#' @details
#' \tabular{ll}{
#' Package: \tab Rdta\cr
#' Type: \tab Package\cr
#' Version: \tab 1.0.0\cr
#' Date: \tab 2020-1-7\cr
#' License: \tab GPL-2\cr
#' Main functions: \tab \code{\link{lmm}}\cr
#' }
#'
#' @author Hyungsuk Tak (maintainer), Kisung You, Sujit K. Ghosh, and Bingyue Su
#'
#' @references
#' \insertRef{tak_data_2019}{Rdta}
#'
#' @docType package
#' @name Rdta
#' @aliases Rdta-package
#' @importFrom stats dnorm rgamma rnorm
#' @import Rdpack
#' @import MCMCpack
#' @import mvtnorm
NULL
