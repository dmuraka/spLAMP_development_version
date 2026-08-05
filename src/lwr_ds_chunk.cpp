// Per-knot inner loop for lwr_ds (downscaling variant).
//
// Compared with the reference lwr_chunk_cpp (spatial linear model on points),
// this version handles the areal-observation downscaling setting:
//   * residuals Resid[k] are defined per area (length N_area)
//   * each point j has a weight a[j] and an area id agg_id[j]
//   * W[k] is an area-level weight (usually 1)
//
// Variance structure:
//   Point-level local model:  resid_j = x_j * b + e_j,  e_j ~ N(0, σ²/w_j),
//   with w_j the kernel weight at point j around the knot. Aggregating with
//   the partition matrix A (a_j on the row for area agg_id[j]), the area-
//   level covariance is A diag(σ²/w) A^T. Because A's rows have disjoint
//   support (each point belongs to exactly one area), this matrix is
//   diagonal with entries V_i = σ² · Σ_{j ∈ area i} a_j² / w_j. The WLS
//   weight per area is therefore W_area_i / V_i (with σ² factored out), and
//   σ² is estimated from the weighted residual sum of squares.
//
// V_i is summed only over training points in the knot's kernel neighborhood
// (which are passed in via nb_id). Kernel pre-filtering in the R caller
// guarantees w_j ≥ 0.05 for any point reaching this kernel, so 1/w_j is
// bounded above by ~20 and the strict-variance form is numerically well-
// behaved without further regularization.
//
// All structural flags (knots_train_only, adaptive bandwidth, c_shrink) are
// handled in R before calling this kernel. This code only loops per knot
// within one chunk, does the areal aggregation via unordered_map, computes
// WXY / WXXW / Sigma / b_sel / bv_sel and accumulates into b_all, bv_inv_all,
// pv_inv_all.

#include <Rcpp.h>
#include <unordered_map>
#include <vector>
#include <cmath>
using namespace Rcpp;

// kernel_id: 1 = exp, 2 = gau
static inline double kfun_scalar(double d, double band, int kernel_id) {
  if (kernel_id == 2) return std::exp(-(d * d) / (band * band));
  return std::exp(-d / band);
}

// [[Rcpp::export]]
void lwr_ds_chunk_cpp(
    List             nb_id,              // List of neighbor indices (1-based) per knot in chunk
    List             nb_dist,            // List of neighbor distances per knot in chunk
    IntegerVector    sel_chunk,          // 1-based row indices into b_old (= knot index)
    NumericVector    local_bands,        // band per knot in chunk
    int              kernel_id,
    int              vc,                 // 1-based column index of x used for spatial process
    NumericVector    resid_area,         // length N_area (areal residual)
    NumericVector    X_area,             // length N_area (areal X[, vc])
    NumericVector    W_area,             // length N_area (areal weight)
    NumericVector    a,                  // length n (point weight)
    IntegerVector    agg_id,             // length n (point → area, 1-based)
    LogicalVector    id_train_flag,      // length n (point is in training area)
    NumericMatrix    x,                  // n x nx (point-level covariates)
    NumericVector    B_var_col,          // length n_knot (B_var[, vc])
    double           c_shrink,           // effective-sample-size shrinkage
    NumericMatrix    b_all,               // n x nx, in-place accumulation
    NumericMatrix    bv_inv_all,          // n x nx, in-place accumulation
    NumericMatrix    pv_inv_all,          // n x nx, in-place accumulation
    NumericMatrix    b_old                // n_knot x nx, in-place write
) {
  const int chunk_sz = sel_chunk.size();
  const int vc0      = vc - 1;
  const double eps   = std::numeric_limits<double>::epsilon();

  // Thread-local scratch buffers (one knot at a time)
  std::vector<double> wei_buf;
  std::unordered_map<int, double> sum_ax_hv;       // Σ a*x (areal design)
  std::unordered_map<int, double> sum_a2_over_w;   // Σ a^2 / w (V_i, strict)

  for (int k = 0; k < chunk_sz; ++k) {
    const int sel_knot = sel_chunk[k] - 1;      // 0-based knot row in b_old
    const double band_k = local_bands[k];

    IntegerVector samp = nb_id[k];
    NumericVector dist = nb_dist[k];
    const int m = samp.size();
    if (m < 3) continue;

    // ---- compute weights, track training-point ESS info
    wei_buf.resize(m);
    int    m_hv          = 0;
    double sum_wei_hv    = 0.0;
    double sum_wei_hv_sq = 0.0;
    sum_ax_hv.clear();
    sum_a2_over_w.clear();

    for (int i = 0; i < m; ++i) {
      const double w    = kfun_scalar(dist[i], band_k, kernel_id);
      wei_buf[i]        = w;
      const int sidx    = samp[i] - 1;
      if (id_train_flag[sidx]) {
        ++m_hv;
        sum_wei_hv    += w;
        sum_wei_hv_sq += w * w;
        const int aid = agg_id[sidx];
        const double aj = a[sidx];
        sum_ax_hv[aid]     += aj * x(sidx, vc0);
        sum_a2_over_w[aid] += (aj * aj) / std::max(w, eps);
      }
    }
    if (m_hv <= 5) continue;
    const int n_train_areas = (int)sum_a2_over_w.size();
    // No explicit n_train_areas constraint: the upstream m_hv > 5 check
    // already enforces enough training points, and empirically allowing
    // knots with only one training area in the kernel window improves
    // accuracy on large/heterogeneous areas without destabilizing WLS.

    // ---- WLS at area level
    //   weight_i = W_area_i / V_i,  V_i = Σ a^2 / w  (strict aggregated variance)
    //   b_sel0   = Σ weight_i * X_sel_i * resid_i / Σ weight_i * X_sel_i^2
    double wxy_sel_csum  = 0.0;
    double wxxw_sel_csum = 0.0;
    for (auto const& kv : sum_a2_over_w) {
      const int    aid       = kv.first;
      const int    area_idx  = aid - 1;
      const double V_i       = kv.second;
      if (V_i <= 0.0) continue;
      const double x_sel_area = sum_ax_hv[aid];
      const double weight_i   = W_area[area_idx] / V_i;
      wxy_sel_csum  += weight_i * x_sel_area * resid_area[area_idx];
      wxxw_sel_csum += weight_i * x_sel_area * x_sel_area;
    }
    if (wxxw_sel_csum <= 0.0) continue;
    const double b_sel0 = wxy_sel_csum / wxxw_sel_csum;
    b_old(sel_knot, vc0) = b_sel0;

    // ---- σ² estimation: weighted residual SS / (n_areas - 1)
    double sigma_sum = 0.0;
    for (auto const& kv : sum_a2_over_w) {
      const int    aid      = kv.first;
      const int    area_idx = aid - 1;
      const double V_i      = kv.second;
      if (V_i <= 0.0) continue;
      const double r_sub = resid_area[area_idx] - X_area[area_idx] * b_sel0;
      sigma_sum += (W_area[area_idx] / V_i) * r_sub * r_sub;
    }
    double sigma = sigma_sum / std::max(1, n_train_areas - 1);
    if (sigma <= 0.0) sigma = eps;

    // ---- ridge via B_var (may be Inf for no ridge)
    const double B_var_s = B_var_col[sel_knot];
    const double lambda  = (std::isinf(B_var_s) || B_var_s <= 0.0)
                            ? 0.0 : sigma / B_var_s;
    const double wxxw_lambda = wxxw_sel_csum + lambda;
    double b_sel   = wxy_sel_csum / wxxw_lambda;
    double bv_sel  = sigma / wxxw_lambda;

    // ---- ESS-based shrinkage (c_shrink > 0)
    if (c_shrink > 0.0) {
      const double n_eff  = (sum_wei_hv * sum_wei_hv) /
                            std::max(sum_wei_hv_sq, eps);
      const double shrink = n_eff / (n_eff + c_shrink);
      b_sel  *= shrink;
      bv_sel *= shrink * shrink;
    }
    if (bv_sel <= 0.0) bv_sel = eps;

    // ---- accumulate onto all neighbor points
    // Predictive variance at point j: σ² * (x_j^2 / WXXW + 1/w_j)
    // since the point-level model is e_j ~ N(0, σ²/w_j).
    const double inv_wxxw   = 1.0 / wxxw_sel_csum;
    const double inv_bv_sel = 1.0 / bv_sel;
    double* b_all_col      = &b_all(0, vc0);
    double* bv_inv_all_col = &bv_inv_all(0, vc0);
    double* pv_inv_all_col = &pv_inv_all(0, vc0);
    const double* x_col    = &x(0, vc0);

    for (int i = 0; i < m; ++i) {
      const int    sidx = samp[i] - 1;
      const double w    = wei_buf[i];
      const double ws   = w * w;
      const double xv   = x_col[sidx];
      const double pv_sel = (xv * xv * inv_wxxw) * sigma
                          + sigma / std::max(w, eps);
      if (pv_sel <= 0.0) continue;
      const double wei2_pv_sel = ws / pv_sel;
      b_all_col[sidx]      += wei2_pv_sel * b_sel;
      bv_inv_all_col[sidx] += wei2_pv_sel * inv_bv_sel;
      pv_inv_all_col[sidx] += wei2_pv_sel;
    }
  }
}
