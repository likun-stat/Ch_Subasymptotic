\name{pnw}
\alias{pnw}
\title{Daily maximum temperature extremes in the Pacific Northwest (PNW)}
\description{
  The \code{pnw} dataset contains summer (JJA) temperature-extremes information for
  441 stations in the Pacific Northwest, covering 1950--2021.
  It includes 10-day block maxima, fitted generalized Pareto parameters at a 95\% threshold,
  probability integral transform (PIT) copula values, station metadata, and location-wise
  GEV fits to seasonal maxima.
}
\format{
  The dataset is a list containing the following elements:
  \describe{
    \item{PNW_JJA_10day}{A numeric matrix of dimension \eqn{441 \times 459} giving
      JJA 10-day block maxima (459 10-day maxima over 1950--2021) at 441 locations.}
    \item{params_GPD}{A numeric matrix of dimension \eqn{441 \times 2} containing
      location-wise generalized Pareto distribution (GPD) parameter estimates fitted above
      the location-specific 95th percentile threshold.}
    \item{U}{A numeric matrix of dimension \eqn{441 \times 459} containing PIT-based
      copula values derived from \code{PNW_JJA_10day} using \code{params_GPD}.}
    \item{stationDF_PNW}{A data frame with 441 rows and 5 columns giving station metadata:
      GHCN station ID, longitude, latitude, elevation, and state (WA, OR, CA, or ID).}
    \item{Loc}{A numeric matrix of dimension \eqn{441 \times 71} of location-parameter estimates
      from location-wise GEV fits to 71 seasonal maxima (1950--2021).}
    \item{Scale}{A numeric matrix of dimension \eqn{441 \times 71} of scale-parameter estimates
      from location-wise GEV fits to 71 seasonal maxima (1950--2021).}
    \item{Shape}{A numeric matrix of dimension \eqn{441 \times 71} of shape-parameter estimates
      from location-wise GEV fits to 71 seasonal maxima (1950--2021).}
  }
}
\details{
  Data are available from GitHub and hence can be gathered using the command
  \code{dataset("pnw")}.
}
\references{
  de Carvalho, M., Huser, R., Naveau, P., and Reich, B. J. (2026).
  \emph{Handbook on Statistics of Extremes}. Chapman & Hall/CRC. Boca Raton, FL.
  
  Zhang, L., Rohrbeck, C., and Opitz, T. (2026).
    Subasymptotic models for spatial extremes.
    In \emph{Handbook on Statistics of Extremes}, Chapter 17, pp. 377–400.
    Chapman & Hall/CRC, Boca Raton, FL.
    
  Menne, M. J., Durre, I., Vose, R. S., Gleason, B. E., & Houston, T. G. (2012). 
  An overview of the global historical climatology network-daily database. 
  \emph{Journal of Atmospheric and Oceanic Technology}, 29(7), 897-910.
}
\source{
  Global Historical Climatology Network-Daily (GHCN-D) database
}
\keyword{GHCN}
\keyword{Heatwave}
