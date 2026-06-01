//! A compact random-forest regressor (no external ML dependency), used to
//! predict `log k_off` from coarse system descriptors.
//!
//! Hand-rolled both to keep the build light (the OOM concern again) and to keep
//! the model fully testable: CART regression trees on bootstrap samples with a
//! random feature subset per split, predictions averaged. Permutation feature
//! importance reports which descriptor drives `k_off`, the interpretable answer
//! her project asks for.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// Forest hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct ForestParams {
    pub n_trees: usize,
    pub max_depth: usize,
    pub min_split: usize,
    /// Features considered per split (`0` => all).
    pub mtry: usize,
}

impl Default for ForestParams {
    fn default() -> Self {
        Self {
            n_trees: 200,
            max_depth: 8,
            min_split: 4,
            mtry: 0,
        }
    }
}

#[derive(Debug, Clone)]
enum Node {
    Leaf(f64),
    Split {
        feature: usize,
        thresh: f64,
        left: usize,
        right: usize,
    },
}

#[derive(Debug, Clone)]
struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    fn predict(&self, x: &[f64]) -> f64 {
        let mut id = 0;
        loop {
            match &self.nodes[id] {
                Node::Leaf(v) => return *v,
                Node::Split {
                    feature,
                    thresh,
                    left,
                    right,
                } => {
                    id = if x[*feature] <= *thresh {
                        *left
                    } else {
                        *right
                    }
                }
            }
        }
    }
}

/// A trained random-forest regressor.
#[derive(Debug, Clone)]
pub struct Forest {
    trees: Vec<Tree>,
    n_features: usize,
}

/// Coefficient of determination `R^2` of predictions against the truth.
#[must_use]
pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len() as f64;
    let mean = y_true.iter().sum::<f64>() / n;
    let ss_tot: f64 = y_true.iter().map(|v| (v - mean).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(t, p)| (t - p).powi(2))
        .sum();
    if ss_tot == 0.0 {
        return if ss_res == 0.0 { 1.0 } else { 0.0 };
    }
    1.0 - ss_res / ss_tot
}

/// A deterministic train/test index split (`test_frac` of `n` held out).
#[must_use]
pub fn train_test_split(n: usize, test_frac: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    idx.shuffle(&mut rng);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let n_test = (n as f64 * test_frac).round() as usize;
    let test = idx[..n_test].to_vec();
    let train = idx[n_test..].to_vec();
    (train, test)
}

fn variance(y: &[f64], idx: &[usize]) -> f64 {
    let n = idx.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = idx.iter().map(|&i| y[i]).sum::<f64>() / n;
    idx.iter().map(|&i| (y[i] - mean).powi(2)).sum::<f64>() / n
}

/// Best `(feature, threshold, left_idx, right_idx)` minimising the split SSE
/// over `mtry` random features, or `None` if no valid split exists.
#[allow(clippy::type_complexity)]
fn best_split<R: Rng>(
    x: &[Vec<f64>],
    y: &[f64],
    idx: &[usize],
    mtry: usize,
    rng: &mut R,
) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
    let n_features = x[0].len();
    let mut features: Vec<usize> = (0..n_features).collect();
    features.shuffle(rng);
    features.truncate(mtry.max(1));

    let mut best: Option<(f64, usize, f64, Vec<usize>, Vec<usize>)> = None;
    for &f in &features {
        let mut vals: Vec<f64> = idx.iter().map(|&i| x[i][f]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals.dedup();
        for w in vals.windows(2) {
            let thresh = 0.5 * (w[0] + w[1]);
            let (l, r): (Vec<usize>, Vec<usize>) = idx.iter().partition(|&&i| x[i][f] <= thresh);
            if l.is_empty() || r.is_empty() {
                continue;
            }
            let sse = variance(y, &l) * l.len() as f64 + variance(y, &r) * r.len() as f64;
            if best.as_ref().is_none_or(|b| sse < b.0) {
                best = Some((sse, f, thresh, l, r));
            }
        }
    }
    best.map(|(_, f, t, l, r)| (f, t, l, r))
}

#[allow(clippy::too_many_arguments)]
fn build_node<R: Rng>(
    x: &[Vec<f64>],
    y: &[f64],
    idx: &[usize],
    depth: usize,
    p: &ForestParams,
    mtry: usize,
    rng: &mut R,
    nodes: &mut Vec<Node>,
) -> usize {
    let id = nodes.len();
    let mean = idx.iter().map(|&i| y[i]).sum::<f64>() / idx.len() as f64;
    nodes.push(Node::Leaf(mean));

    if depth >= p.max_depth || idx.len() < p.min_split {
        return id;
    }
    if let Some((feature, thresh, left_idx, right_idx)) = best_split(x, y, idx, mtry, rng) {
        let left = build_node(x, y, &left_idx, depth + 1, p, mtry, rng, nodes);
        let right = build_node(x, y, &right_idx, depth + 1, p, mtry, rng, nodes);
        nodes[id] = Node::Split {
            feature,
            thresh,
            left,
            right,
        };
    }
    id
}

impl Forest {
    /// Fit a forest to rows `x` with targets `y`.
    #[must_use]
    pub fn fit(x: &[Vec<f64>], y: &[f64], p: &ForestParams, seed: u64) -> Forest {
        let n = x.len();
        let n_features = x.first().map_or(0, Vec::len);
        let mtry = if p.mtry == 0 {
            n_features
        } else {
            p.mtry.min(n_features)
        };
        let trees = (0..p.n_trees)
            .map(|t| {
                let mut rng =
                    StdRng::seed_from_u64(seed ^ (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                // bootstrap sample (with replacement)
                let boot: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
                let mut nodes = Vec::new();
                build_node(x, y, &boot, 0, p, mtry, &mut rng, &mut nodes);
                Tree { nodes }
            })
            .collect();
        Forest { trees, n_features }
    }

    /// Predict the target for one feature row.
    #[must_use]
    pub fn predict(&self, x: &[f64]) -> f64 {
        if self.trees.is_empty() {
            return 0.0;
        }
        self.trees.iter().map(|t| t.predict(x)).sum::<f64>() / self.trees.len() as f64
    }

    /// Predict for many rows.
    #[must_use]
    pub fn predict_many(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|row| self.predict(row)).collect()
    }
}

/// Permutation feature importance: the drop in `R^2` when each feature column is
/// shuffled. Larger means the model relies on that descriptor more.
#[must_use]
pub fn permutation_importance(forest: &Forest, x: &[Vec<f64>], y: &[f64], seed: u64) -> Vec<f64> {
    let base = r2_score(y, &forest.predict_many(x));
    let mut rng = StdRng::seed_from_u64(seed);
    (0..forest.n_features)
        .map(|f| {
            let mut col: Vec<f64> = x.iter().map(|r| r[f]).collect();
            col.shuffle(&mut rng);
            let mut xp = x.to_vec();
            for (row, &v) in xp.iter_mut().zip(col.iter()) {
                row[f] = v;
            }
            base - r2_score(y, &forest.predict_many(&xp))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r2_is_one_for_perfect_and_zero_for_mean() {
        let y = [1.0, 2.0, 3.0, 4.0];
        assert!((r2_score(&y, &y) - 1.0).abs() < 1e-12);
        let mean = 2.5;
        let mean_pred = [mean; 4];
        assert!(r2_score(&y, &mean_pred).abs() < 1e-12);
    }

    #[test]
    fn split_holds_out_requested_fraction() {
        let (train, test) = train_test_split(100, 0.3, 1);
        assert_eq!(test.len(), 30);
        assert_eq!(train.len(), 70);
        let mut all: Vec<usize> = train.iter().chain(test.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..100).collect::<Vec<_>>());
    }

    fn make_data(n: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..n {
            let a = rng.gen_range(0.0..5.0);
            let b = rng.gen_range(0.0..5.0);
            let noise = rng.gen_range(-0.1..0.1);
            x.push(vec![a, b]);
            y.push(2.0 * a - b + noise); // depends mostly on feature 0
        }
        (x, y)
    }

    #[test]
    fn forest_learns_a_smooth_function() {
        let (xtr, ytr) = make_data(300, 1);
        let (xte, yte) = make_data(120, 2);
        let f = Forest::fit(&xtr, &ytr, &ForestParams::default(), 7);
        let pred = f.predict_many(&xte);
        let r2 = r2_score(&yte, &pred);
        assert!(
            r2 > 0.9,
            "forest should fit the smooth function, R^2={r2:.3}"
        );
    }

    #[test]
    fn importance_ranks_the_driving_feature_first() {
        let (x, y) = make_data(400, 3);
        let f = Forest::fit(&x, &y, &ForestParams::default(), 7);
        let imp = permutation_importance(&f, &x, &y, 11);
        assert!(
            imp[0] > imp[1],
            "feature 0 (coeff 2) should matter more than feature 1 (coeff -1): {imp:?}"
        );
    }
}
