// Fused serial kernel for lwr_glm: builds a nanoflann kd-tree over `coords`
// (and over `coords0` when prediction sites are given) ONCE, then for each
// knot does (radius search -> local GLM -> scatter-add) inline, WITHOUT
// materialising the neighbour lists in R. Numerically identical to the
// frNN + lwr_chunk_glm_cpp path (radius search finds the same neighbours;
// tiny ~1e-13 differences only from sqrt(squared-dist) vs direct distance).
// Much lower peak memory (no ~O(N*avg_nb) neighbour list) and a bit faster.
// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>
#include "nanoflann.h"
#include <vector>
#include <cmath>
#include <cstdint>
using namespace Rcpp;

struct PC2 {
  const double* pts; std::size_t N;
  inline std::size_t kdtree_get_point_count() const { return N; }
  inline double kdtree_get_pt(std::size_t i, std::size_t d) const { return pts[i + d * N]; }
  template<class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};
typedef nanoflann::KDTreeSingleIndexAdaptor<
          nanoflann::L2_Simple_Adaptor<double, PC2>, PC2, 2> KDTree2;

static inline double kfun_f(double d, double band, int kid) {
  if (kid == 2) return std::exp(-(d * d) / (band * band));
  return std::exp(-d / band);
}

// [[Rcpp::export]]
List lwr_glm_fused_cpp(
    NumericMatrix coords,        // n x 2  (training points; tree)
    NumericMatrix coords_cent,   // n_knot x 2 (knot query locations)
    NumericVector resid,         // n
    NumericVector w_obs,         // n
    NumericMatrix x,             // n x nx
    IntegerVector id_train,      // n (0/1)
    NumericMatrix B_var,         // n_knot x nx
    IntegerVector vc_cols,       // 1-based columns with a varying coefficient
    double band, int kernel_id, double threshold, int is_lm,
    SEXP coords0_sexp,           // n0 x 2 or NULL
    SEXP x0_sexp) {              // n0 x nx or NULL

  const int n      = x.nrow();
  const int nx     = x.ncol();
  const int n_knot = coords_cent.nrow();
  const int n_vc   = vc_cols.size();
  const bool has0  = !Rf_isNull(coords0_sexp);
  const double rad2 = threshold * threshold;

  const double* cp   = coords.begin();
  const double* cc   = coords_cent.begin();
  const double* xp   = x.begin();
  const double* rp   = resid.begin();
  const double* wp   = w_obs.begin();
  const int*    idt  = id_train.begin();
  const double* Bv   = B_var.begin();
  const int*    vcp  = vc_cols.begin();

  PC2 cloud{cp, (std::size_t)n};
  KDTree2 tree(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  tree.buildIndex();

  int n0 = 0;
  NumericMatrix x0_mat;
  const double* x0p = 0;
  std::vector<double> c0buf;          // owns coords0 for the tree's lifetime
  PC2 cloud0{0, 0};
  KDTree2* tree0 = 0;
  if (has0) {
    NumericMatrix c0(coords0_sexp);
    x0_mat = NumericMatrix(x0_sexp);
    n0 = c0.nrow();
    c0buf.assign(c0.begin(), c0.end());
    cloud0.pts = c0buf.data(); cloud0.N = (std::size_t)n0;
    tree0 = new KDTree2(2, cloud0, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree0->buildIndex();
    x0p = x0_mat.begin();
  }

  NumericMatrix b_all(n, nx), bv_inv_all(n, nx), pv_inv_all(n, nx), b_old(n_knot, nx);
  NumericMatrix b_all0, bv_inv_all0, pv_inv_all0;
  if (has0) { b_all0 = NumericMatrix(n0, nx); bv_inv_all0 = NumericMatrix(n0, nx); pv_inv_all0 = NumericMatrix(n0, nx); }
  double* ba = b_all.begin(); double* bvv = bv_inv_all.begin(); double* pvv = pv_inv_all.begin(); double* bo = b_old.begin();
  double* ba0 = has0 ? b_all0.begin() : 0; double* bv0 = has0 ? bv_inv_all0.begin() : 0; double* pv0 = has0 ? pv_inv_all0.begin() : 0;

  std::vector<nanoflann::ResultItem<uint32_t, double> > mt, mt0;
  std::vector<double> wei, wei0;
  nanoflann::SearchParameters sprm; sprm.sorted = false;

  for (int k = 0; k < n_knot; ++k) {
    double q[2] = { cc[k], cc[k + (std::size_t)n_knot] };
    mt.clear();
    const std::size_t m = tree.radiusSearch(q, rad2, mt, sprm);
    if (m == 0) continue;
    if (wei.size() < m) wei.resize(m);

    int m_hv = 0; double wxy = 0.0, wxxw = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
      const double wker = kfun_f(std::sqrt(mt[i].second), band, kernel_id);
      wei[i] = wker;
      const int sidx = (int)mt[i].first;
      if (idt[sidx]) {
        const double ww = wker * wker, w_o = wp[sidx];
        wxy += ww * w_o * rp[sidx]; wxxw += ww * w_o; ++m_hv;
      }
    }
    if (m_hv <= 5 || wxxw <= 0.0) continue;
    const double b_sel0 = wxy / wxxw;
    for (int j = 0; j < nx; ++j) bo[k + (std::size_t)j * n_knot] = b_sel0;

    std::size_t m0 = 0;
    if (has0) {
      mt0.clear();
      m0 = tree0->radiusSearch(q, rad2, mt0, sprm);
      if (m0 > 0) {
        if (wei0.size() < m0) wei0.resize(m0);
        for (std::size_t i = 0; i < m0; ++i)
          wei0[i] = kfun_f(std::sqrt(mt0[i].second), band, kernel_id);
      }
    }

    for (int vi = 0; vi < n_vc; ++vi) {
      const int j = vcp[vi] - 1;
      const double* xj = xp + (std::size_t)j * n;
      double sigma = 0.0;
      if (is_lm) {
        // LM kernel: variance over ALL neighbours (train+test), unweighted,
        // divided by (m - 1). (m > m_hv > 5, so m - 1 >= 5.)
        for (std::size_t i = 0; i < m; ++i) {
          const int sidx = (int)mt[i].first;
          const double rs = rp[sidx] - xj[sidx] * b_sel0;
          const double v  = wei[i] * rs;
          sigma += v * v;
        }
        sigma /= (m - 1);
      } else {
        // GLM kernel: variance over TRAIN neighbours only, IRLS-weighted,
        // divided by (m_hv - 1).
        for (std::size_t i = 0; i < m; ++i) {
          const int sidx = (int)mt[i].first;
          if (!idt[sidx]) continue;
          const double rs = rp[sidx] - xj[sidx] * b_sel0;
          const double v  = wei[i] * rs;
          sigma += wp[sidx] * v * v;
        }
        if (m_hv <= 1) continue;
        sigma /= (m_hv - 1);
      }
      const double lambda    = sigma / Bv[k + (std::size_t)j * n_knot];
      const double wxxw_lam  = wxxw + lambda;
      const double b_sel_val = wxy / wxxw_lam;
      const double bv_sel    = sigma / wxxw_lam;
      const double inv_bv    = 1.0 / bv_sel;
      const double inv_wxxw  = 1.0 / wxxw;
      double* baj = ba + (std::size_t)j * n; double* bvj = bvv + (std::size_t)j * n; double* pvj = pvv + (std::size_t)j * n;
      for (std::size_t i = 0; i < m; ++i) {
        const int sidx = (int)mt[i].first;
        const double wk = wei[i], ws = wk * wk, xv = xj[sidx];
        const double pv_sel = (xv * xv * inv_wxxw) * sigma + sigma / wk;
        const double wei2   = ws / pv_sel;
        baj[sidx] += wei2 * b_sel_val; bvj[sidx] += wei2 * inv_bv; pvj[sidx] += wei2;
      }
      if (has0 && m0 > 0) {
        const double* x0j = x0p + (std::size_t)j * n0;
        double* ba0j = ba0 + (std::size_t)j * n0; double* bv0j = bv0 + (std::size_t)j * n0; double* pv0j = pv0 + (std::size_t)j * n0;
        for (std::size_t i = 0; i < m0; ++i) {
          const int sidx0 = (int)mt0[i].first;
          const double w0 = wei0[i], w0s = w0 * w0, xv0 = x0j[sidx0];
          const double pv_sel0 = (xv0 * xv0 * inv_wxxw) * sigma + sigma / w0;
          const double wei2_0  = w0s / pv_sel0;
          ba0j[sidx0] += wei2_0 * b_sel_val; bv0j[sidx0] += wei2_0 * inv_bv; pv0j[sidx0] += wei2_0;
        }
      }
    }
  }
  if (tree0) delete tree0;

  if (has0)
    return List::create(_["b_all"]=b_all,_["bv_inv_all"]=bv_inv_all,_["pv_inv_all"]=pv_inv_all,
                        _["b_old"]=b_old,_["b_all0"]=b_all0,_["bv_inv_all0"]=bv_inv_all0,_["pv_inv_all0"]=pv_inv_all0);
  return List::create(_["b_all"]=b_all,_["bv_inv_all"]=bv_inv_all,_["pv_inv_all"]=pv_inv_all,_["b_old"]=b_old);
}
