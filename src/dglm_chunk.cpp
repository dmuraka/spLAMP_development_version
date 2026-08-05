// Fused per-scale operator for CF-DGLMM: neighbour-limited kernel aggregation
// -> per-knot AR(1) Kalman filter + RTS smoother -> generalized Product-of-
// Experts (gPoE) recombination, in a single pass over the sparse neighbour
// structure. Replaces the chain of Matrix sparse products + .dglm_ksmooth,
// avoiding intermediate K x T matrices and repeated sparse traversals.
//
// LAYOUT: every hot array is stored time-major (T is the leading/contiguous
// dimension) so the innermost t-loop runs over contiguous memory. The working
// panels arrive transposed as W0t, R0t (T x nL, column-major) so W0t[t + T*i]
// is contiguous in t. Aggregates/state/gPoE accumulators are T x K or T x n.
//
// Neighbour lists are CSR: for site i, its knots are idx[ptr[i] .. ptr[i+1]-1]
// with kernel weights w[...] (0-based knot indices).
#include <Rcpp.h>
#include <vector>
using namespace Rcpp;

// [[Rcpp::export]]
List dglm_scale_chunk(IntegerVector ptr, IntegerVector idx, NumericVector w,
                      NumericMatrix W0t, NumericMatrix R0t,
                      int K, double rho, double Q,
                      IntegerVector pptr, IntegerVector pidx, NumericVector pw,
                      int n0) {
  const int T  = W0t.nrow();          // time-major: rows = time
  const int nL = W0t.ncol();          //            cols = sites
  const double eps = 1e-12;
  const double *pW0 = &W0t[0], *pR0 = &R0t[0];

  // ---- 1. aggregate working residual to knots: den, Znum, Rnum (T x K) ----
  std::vector<double> den((size_t)T * K, 0.0), Znum((size_t)T * K, 0.0),
                      Rnum((size_t)T * K, 0.0);
  for (int i = 0; i < nL; ++i) {
    const double *w0i = pW0 + (size_t)T * i;
    const double *r0i = pR0 + (size_t)T * i;
    for (int nz = ptr[i]; nz < ptr[i + 1]; ++nz) {
      const int k = idx[nz];
      const double wik = w[nz], wik2 = wik * wik;
      double *dk = &den[(size_t)T * k], *zk = &Znum[(size_t)T * k],
             *rk = &Rnum[(size_t)T * k];
      for (int t = 0; t < T; ++t) {
        const double w0 = w0i[t];
        if (w0 == 0.0) continue;
        const double wW = wik * w0;
        dk[t] += wW;
        zk[t] += wW * r0i[t];
        rk[t] += wik2 * w0;
      }
    }
  }

  // ---- 2. per-knot AR(1) Kalman filter + RTS smoother -> m, P (T x K) ----
  std::vector<double> m((size_t)T * K), P((size_t)T * K);
  std::vector<double> af(T), Pf(T), ap(T), Pp(T);
  const double P0 = Q / (1.0 - rho * rho);
  for (int k = 0; k < K; ++k) {
    const double *dk = &den[(size_t)T * k], *zk = &Znum[(size_t)T * k],
                 *rk = &Rnum[(size_t)T * k];
    double *mk = &m[(size_t)T * k], *Pk = &P[(size_t)T * k];
    double a = 0.0, p = P0;
    for (int t = 0; t < T; ++t) {
      const double a_pred = rho * a;
      const double p_pred = rho * rho * p + Q;
      ap[t] = a_pred; Pp[t] = p_pred;
      const double d = dk[t];
      if (d > eps) {                                  // observed knot-time -> update
        const double Zkt = zk[t] / d;
        const double Rkt = rk[t] / (d * d);
        const double Kg  = p_pred / (p_pred + Rkt);
        a = a_pred + Kg * (Zkt - a_pred);
        p = (1.0 - Kg) * p_pred;
      } else { a = a_pred; p = p_pred; }              // missing -> predict only
      af[t] = a; Pf[t] = p;
    }
    double ms = af[T - 1], ps = Pf[T - 1];
    mk[T - 1] = ms; Pk[T - 1] = ps > 1e-8 ? ps : 1e-8;
    for (int t = T - 2; t >= 0; --t) {
      double pp1 = Pp[t + 1]; if (pp1 < 1e-12) pp1 = 1e-12;
      const double G = rho * Pf[t] / pp1;
      ms = af[t] + G * (ms - ap[t + 1]);
      ps = Pf[t] + G * G * (ps - Pp[t + 1]);
      mk[t] = ms; Pk[t] = ps > 1e-8 ? ps : 1e-8;
    }
  }
  // precompute knot precision and mean*precision (T x K) for gPoE
  std::vector<double> invP((size_t)T * K), mP((size_t)T * K);
  for (size_t j = 0; j < invP.size(); ++j) { invP[j] = 1.0 / P[j]; mP[j] = m[j] * invP[j]; }

  // ---- 3. gPoE recombination over a set of sites' CSR neighbours ----
  // Generalized product of experts with kernel weights NORMALIZED to sum to one
  // (Cao & Fleet 2014): with k~_ik = w_ik / sum_k w_ik,
  //   F(i,t) = sum_k (w_ik/P_kt) m_kt / sum_k (w_ik/P_kt)   (normalizer cancels),
  //   V(i,t) = 1 / sum_k (k~_ik/P_kt) = (sum_k w_ik) / sum_k (w_ik/P_kt).
  // The mean is unchanged versus the unnormalized form; only the variance differs
  // (it no longer shrinks purely with the number of nearby knots).
  auto gpoe = [&](IntegerVector P_ptr, IntegerVector P_idx, NumericVector P_w,
                  int n, NumericMatrix &F, NumericMatrix &V) {
    std::vector<double> gden((size_t)T * n, 0.0), gnum((size_t)T * n, 0.0),
                        sumw((size_t)n, 0.0);
    for (int i = 0; i < n; ++i) {
      double *gd = &gden[(size_t)T * i], *gn = &gnum[(size_t)T * i];
      double sw = 0.0;
      for (int nz = P_ptr[i]; nz < P_ptr[i + 1]; ++nz) {
        const int k = P_idx[nz];
        const double wik = P_w[nz];
        sw += wik;                              // sum of kernel weights (t-independent)
        const double *iPk = &invP[(size_t)T * k], *mPk = &mP[(size_t)T * k];
        for (int t = 0; t < T; ++t) { gd[t] += wik * iPk[t]; gn[t] += wik * mPk[t]; }
      }
      sumw[i] = sw;
    }
    double vmx = 1.0;
    for (int i = 0; i < n; ++i)
      for (int t = 0; t < T; ++t) {
        const double g = gden[(size_t)T * i + t];
        if (g > 0.0) { double v = sumw[i] / g; if (v > vmx) vmx = v; }
      }
    for (int i = 0; i < n; ++i) {
      const double *gd = &gden[(size_t)T * i], *gn = &gnum[(size_t)T * i];
      const double sw = sumw[i];
      for (int t = 0; t < T; ++t) {
        const double g = gd[t];
        if (g > 0.0) { F(i, t) = gn[t] / g; V(i, t) = sw / g; }
        else         { F(i, t) = 0.0;       V(i, t) = vmx;    }
      }
    }
  };

  NumericMatrix Ftr(nL, T), Vtr(nL, T);
  gpoe(ptr, idx, w, nL, Ftr, Vtr);
  List out = List::create(_["Ftr"] = Ftr, _["Vtr"] = Vtr);
  if (n0 > 0) {
    NumericMatrix Fpr(n0, T), Vpr(n0, T);
    gpoe(pptr, pidx, pw, n0, Fpr, Vpr);
    out["Fpr"] = Fpr; out["Vpr"] = Vpr;
  }
  return out;
}
