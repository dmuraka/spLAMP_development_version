#include <Rcpp.h>
using namespace Rcpp;

// kernel_id: 1 = exp, 2 = gau
static inline double kfun_scalar(double d, double band, int kernel_id) {
  if (kernel_id == 2) return std::exp(-(d * d) / (band * band));
  return std::exp(-d / band);
}

// [[Rcpp::export]]
void lwr_chunk_cpp(
    List             nb_id,
    List             nb_dist,
    SEXP             nb_id0_sexp,
    SEXP             nb_dist0_sexp,
    IntegerVector    sel_chunk,        // 1-based row indices into b_old
    LogicalVector    id_train_flag,    // length n
    NumericVector    resid,            // length n
    NumericMatrix    x,                // n x nx
    SEXP             x0_sexp,          // NumericMatrix or NULL
    NumericMatrix    B_var,            // n_knot x nx
    IntegerVector    vc_cols,          // 1-based columns to process
    double           band,
    int              kernel_id,
    NumericMatrix    b_all,            // n x nx, accumulated in place
    NumericMatrix    bv_inv_all,       // n x nx, accumulated in place
    NumericMatrix    pv_inv_all,       // n x nx, accumulated in place
    SEXP             b_all0_sexp,      // optional
    SEXP             bv_inv_all0_sexp, // optional
    SEXP             pv_inv_all0_sexp, // optional
    NumericMatrix    b_old             // n_knot x nx, filled in place
) {
  const int nx       = x.ncol();
  const int chunk_sz = sel_chunk.size();
  const int n_vc     = vc_cols.size();
  const bool has0    = !Rf_isNull(nb_id0_sexp);

  List          nb_id0, nb_dist0;
  NumericMatrix x0_mat, b_all0_mat, bv_inv_all0_mat, pv_inv_all0_mat;
  if (has0) {
    nb_id0           = as<List>(nb_id0_sexp);
    nb_dist0         = as<List>(nb_dist0_sexp);
    x0_mat           = as<NumericMatrix>(x0_sexp);
    b_all0_mat       = as<NumericMatrix>(b_all0_sexp);
    bv_inv_all0_mat  = as<NumericMatrix>(bv_inv_all0_sexp);
    pv_inv_all0_mat  = as<NumericMatrix>(pv_inv_all0_sexp);
  }

  std::vector<double> wei_buf;
  std::vector<double> wei0_buf;

  for (int k = 0; k < chunk_sz; ++k) {
    const int sel = sel_chunk[k] - 1;

    IntegerVector samp = nb_id[k];
    NumericVector dist = nb_dist[k];
    const int m = samp.size();
    if (m == 0) continue;

    wei_buf.resize(m);

    int    m_hv          = 0;
    double wxy_sel_csum  = 0.0;
    double wxxw_sel_csum = 0.0;

    for (int i = 0; i < m; ++i) {
      const double w = kfun_scalar(dist[i], band, kernel_id);
      wei_buf[i] = w;
      const int sidx = samp[i] - 1;
      if (id_train_flag[sidx]) {
        const double ww = w * w;
        wxy_sel_csum  += ww * resid[sidx];
        wxxw_sel_csum += ww;
        ++m_hv;
      }
    }
    if (m_hv <= 5) continue;
    if (wxxw_sel_csum <= 0.0) continue;

    const double b_sel0 = wxy_sel_csum / wxxw_sel_csum;
    for (int j = 0; j < nx; ++j) {
      b_old(sel, j) = b_sel0;
    }

    // coords0 neighbors (optional)
    int           m0 = 0;
    IntegerVector samp0;
    NumericVector dist0;
    if (has0) {
      samp0 = nb_id0[k];
      dist0 = nb_dist0[k];
      m0    = samp0.size();
      if (m0 > 0) {
        wei0_buf.resize(m0);
        for (int i = 0; i < m0; ++i) {
          wei0_buf[i] = kfun_scalar(dist0[i], band, kernel_id);
        }
      }
    }

    for (int vi = 0; vi < n_vc; ++vi) {
      const int j = vc_cols[vi] - 1;

      // Column-pointer caching: avoid recomputing j*nrow() for every element
      // access inside the per-neighbor inner loops (option 4 micro-opt).
      const double* xj           = &x(0, j);
      double*       b_all_j      = &b_all(0, j);
      double*       bv_inv_all_j = &bv_inv_all(0, j);
      double*       pv_inv_all_j = &pv_inv_all(0, j);

      // sigma = sum( (wei * (resid[samp] - x[samp,j]*b_sel0))^2 ) / (m - 1)
      double sigma = 0.0;
      for (int i = 0; i < m; ++i) {
        const int sidx = samp[i] - 1;
        const double rs = resid[sidx] - xj[sidx] * b_sel0;
        const double v  = wei_buf[i] * rs;
        sigma += v * v;
      }
      sigma /= (m - 1);

      const double B_var_sj   = B_var(sel, j);
      const double lambda     = sigma / B_var_sj;       // Inf → 0
      const double wxxw_lam   = wxxw_sel_csum + lambda;
      const double b_sel_val  = wxy_sel_csum / wxxw_lam;
      const double bv_sel     = sigma / wxxw_lam;
      const double inv_bv_sel = 1.0 / bv_sel;
      const double inv_wxxw   = 1.0 / wxxw_sel_csum;

      for (int i = 0; i < m; ++i) {
        const int sidx = samp[i] - 1;
        const double w  = wei_buf[i];
        const double ws = w * w;
        const double xv = xj[sidx];
        const double pv_sel      = (xv * xv * inv_wxxw) * sigma + sigma / w;
        const double wei2_pv_sel = ws / pv_sel;
        b_all_j[sidx]      += wei2_pv_sel * b_sel_val;
        bv_inv_all_j[sidx] += wei2_pv_sel * inv_bv_sel;
        pv_inv_all_j[sidx] += wei2_pv_sel;
      }

      if (has0 && m0 > 0) {
        const double* x0_j           = &x0_mat(0, j);
        double*       b_all0_j       = &b_all0_mat(0, j);
        double*       bv_inv_all0_j  = &bv_inv_all0_mat(0, j);
        double*       pv_inv_all0_j  = &pv_inv_all0_mat(0, j);
        for (int i = 0; i < m0; ++i) {
          const int sidx0 = samp0[i] - 1;
          const double w0  = wei0_buf[i];
          const double w0s = w0 * w0;
          const double xv0 = x0_j[sidx0];
          const double pv_sel0      = (xv0 * xv0 * inv_wxxw) * sigma + sigma / w0;
          const double wei2_pv_sel0 = w0s / pv_sel0;
          b_all0_j[sidx0]      += wei2_pv_sel0 * b_sel_val;
          bv_inv_all0_j[sidx0] += wei2_pv_sel0 * inv_bv_sel;
          pv_inv_all0_j[sidx0] += wei2_pv_sel0;
        }
      }
    }
  }
}
