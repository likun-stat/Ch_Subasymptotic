#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <R_ext/Lapack.h>
#include <R_ext/Utils.h>

void pmvnormE(double *bounds, int *n, double *cor, double *prob, double *sd,
	     int *N, int *M){
  /* This function computes the CDF of a standard multivariate normal
     random vector using quasi monte carlo sequence*/

  const int primeNumbers[100] = {
    2,   3,   5,   7,  11,  13,  17,  19,  23,  29,
    31,  37,  41,  43,  47,  53,  59,  61,  67,  71,
    73,  79,  83,  89,  97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
    353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
    467, 479, 487, 491, 499, 503, 509, 521, 523, 541};

  // 1. We need to reorder the variables
  double *newCor = malloc(*n * *n * sizeof(double));
  memcpy(newCor, cor, *n * *n * sizeof(double));

  int *idx = malloc(*n * sizeof(int));
  for (int i=0;i<*n;i++)
    idx[i] = i;

  rsort_with_index(bounds, idx, *n);

  for (int i=0;i<*n;i++)
    for (int j=0;j<*n;j++)
      newCor[i + j * *n] = cor[idx[i] + *n * idx[j]];

  //Compute the Cholesky decomposition of newCor
  int info = 0;
  F77_CALL(dpotrf)("L", n, newCor, n, &info);

  if (info != 0)
    error("1. error code %d from Lapack routine '%s'", info, "dpotrf");

  // Get the inverse entries of newCor for efficiency
  double *inewCor = malloc(*n * *n * sizeof(double));
  for (int i=0;i<*n * *n;i++)
    inewCor[i] = 1 / newCor[i];

  // Initialize objects
  double T = 0, V = 0;
  double *I = malloc(*M * sizeof(double)),
    *Delta = malloc(*n * sizeof(double)),
    delta,
    *q = malloc(*n * sizeof(double)),
    *e = malloc(*n * sizeof(double)),
    *f = malloc(*n * sizeof(double)),
    *w = malloc(*n * sizeof(double)),
    *y = malloc((*n - 1) * sizeof(double)),
    *ea = malloc(*n * sizeof(double)),
    *fa = malloc(*n * sizeof(double)),
    *wa = malloc(*n * sizeof(double)),
    *ya = malloc((*n - 1) * sizeof(double));//a for antithethic

  for (int i=0;i<*n;i++)
    q[i] = sqrt(primeNumbers[i]);

  e[0] = ea[0] = pnorm(bounds[0] * inewCor[0], 0, 1, 1, 0);
  f[0] = fa[0] = e[0];

  GetRNGstate();

  for (int i=0;i<*M;i++){
    I[i] = 0;

    for (int j=0;j<*n;j++)
      Delta[j] = unif_rand();

    for (int j=0;j<*N;j++){
      for (int k=0;k<*n;k++){
	double dummy = j * q[k] + Delta[k];
	w[k] = fabs(2 * (dummy - ftrunc(dummy)) - 1);
	wa[k] = 1 - w[k];
      }

      for (int k=1;k<*n;k++){
	y[k-1] = qnorm(w[k-1] * e[k-1], 0, 1, 1, 0);
	ya[k-1] = qnorm(wa[k-1] * ea[k-1], 0, 1, 1, 0);

	double dummy = 0, dummya = 0;
	for (int l=0;l<k;l++){
	  dummy += newCor[k + l * *n] * y[l];
	  dummya += newCor[k + l * *n] * ya[l];
	}
	
	e[k] = pnorm((bounds[k] - dummy) * inewCor[k * (*n + 1)], 0, 1, 1, 0);
	ea[k] = pnorm((bounds[k] - dummya) * inewCor[k * (*n + 1)], 0, 1, 1, 0);
	f[k] = e[k] * f[k - 1];
	fa[k] = ea[k] * fa[k - 1];
      }

      I[i] = I[i] + (0.5 * (f[*n - 1] + fa[*n - 1]) - I[i]) / ((double) j + 1);
    }

    delta = (I[i] - T) / ((double) i + 1);
    T += delta;
    V = (i - 1) * V / ((double) i + 1) + delta * delta;
  }

  PutRNGstate();

  *prob = T;
  *sd = sqrt(V);

  free(newCor); free(idx); free(inewCor); free(I); free(Delta);
  free(q); free(e); free(f); free(w); free(y); free(ea); free(fa);
  free(wa); free(ya);

  return;
}