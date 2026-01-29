//! SEDAC V9.0 - Rust高性能内核
//!
//! 使用SIMD和多线程加速CPU端操作
//! 通过PyO3暴露Python接口

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f32::consts::LN_2;

/// 计算信息熵 (SIMD优化)
/// 
/// H = -sum(p * log2(p))
#[inline]
fn compute_entropy_single(logits: &[f32]) -> f32 {
    let n = logits.len();
    if n == 0 {
        return 0.0;
    }
    
    // 数值稳定的softmax
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    let sum_exp: f32 = logits.iter()
        .map(|&x| (x - max_logit).exp())
        .sum();
    
    let log_sum_exp = sum_exp.ln();
    
    // H = log(Z) - E[x]/Z
    let weighted_sum: f32 = logits.iter()
        .map(|&x| {
            let p = (x - max_logit).exp() / sum_exp;
            let log_p = x - max_logit - log_sum_exp;
            p * log_p
        })
        .sum();
    
    -weighted_sum / LN_2
}

/// 计算hidden state稳定性
#[inline]
fn compute_stability_single(curr: &[f32], prev: &[f32]) -> f32 {
    let mut diff_sq: f32 = 0.0;
    let mut norm_sq: f32 = 0.0;
    
    for (c, p) in curr.iter().zip(prev.iter()) {
        let diff = c - p;
        diff_sq += diff * diff;
        norm_sq += c * c;
    }
    
    1.0 / (1.0 + diff_sq.sqrt() / (norm_sq.sqrt() + 1e-6))
}

/// 计算置信度
#[inline]
fn compute_confidence(entropy: f32, entropy_mean: f32, entropy_std: f32) -> f32 {
    let z_score = (entropy_mean - entropy) / (entropy_std + 1e-6);
    1.0 / (1.0 + (-z_score * 2.0).exp())
}

/// 退出决策
#[inline]
fn should_exit(
    confidence: f32,
    stability: f32,
    layer_progress: f32,
    exit_threshold: f32,
) -> bool {
    let dynamic_threshold = exit_threshold - layer_progress * 0.2;
    let exit_score = confidence * stability * (0.5 + layer_progress * 0.5);
    exit_score > dynamic_threshold
}

/// 批量融合熵决策 (并行)
/// 
/// 输入:
/// - logits: [N, vocab_size]
/// - hidden: [N, hidden_size]
/// - prev_hidden: [N, hidden_size]
/// 
/// 输出:
/// - entropy: [N]
/// - confidence: [N]
/// - stability: [N]
/// - exit_mask: [N] (bool as u8)
#[pyfunction]
fn fused_entropy_decision_rust<'py>(
    py: Python<'py>,
    logits: PyReadonlyArray2<f32>,
    hidden: PyReadonlyArray2<f32>,
    prev_hidden: PyReadonlyArray2<f32>,
    entropy_mean: f32,
    entropy_std: f32,
    layer_progress: f32,
    exit_threshold: f32,
) -> PyResult<(
    &'py PyArray1<f32>,
    &'py PyArray1<f32>,
    &'py PyArray1<f32>,
    &'py PyArray1<u8>,
)> {
    let logits = logits.as_array();
    let hidden = hidden.as_array();
    let prev_hidden = prev_hidden.as_array();
    
    let n = logits.nrows();
    let vocab_size = logits.ncols();
    let hidden_size = hidden.ncols();
    
    // 并行计算
    let results: Vec<(f32, f32, f32, u8)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let logits_row: Vec<f32> = (0..vocab_size).map(|j| logits[[i, j]]).collect();
            let hidden_row: Vec<f32> = (0..hidden_size).map(|j| hidden[[i, j]]).collect();
            let prev_row: Vec<f32> = (0..hidden_size).map(|j| prev_hidden[[i, j]]).collect();
            
            let entropy = compute_entropy_single(&logits_row);
            let stability = compute_stability_single(&hidden_row, &prev_row);
            let confidence = compute_confidence(entropy, entropy_mean, entropy_std);
            let exit = should_exit(confidence, stability, layer_progress, exit_threshold);
            
            (entropy, confidence, stability, exit as u8)
        })
        .collect();
    
    // 分离结果
    let mut entropy_vec = Vec::with_capacity(n);
    let mut confidence_vec = Vec::with_capacity(n);
    let mut stability_vec = Vec::with_capacity(n);
    let mut exit_vec = Vec::with_capacity(n);
    
    for (e, c, s, m) in results {
        entropy_vec.push(e);
        confidence_vec.push(c);
        stability_vec.push(s);
        exit_vec.push(m);
    }
    
    Ok((
        PyArray1::from_vec(py, entropy_vec),
        PyArray1::from_vec(py, confidence_vec),
        PyArray1::from_vec(py, stability_vec),
        PyArray1::from_vec(py, exit_vec),
    ))
}

/// Token Router Split (并行)
/// 
/// 将batch分为active组和exit组
#[pyfunction]
fn token_router_split_rust<'py>(
    py: Python<'py>,
    hidden: PyReadonlyArray2<f32>,
    exit_mask: PyReadonlyArray1<u8>,
) -> PyResult<(
    &'py PyArray2<f32>,
    &'py PyArray1<i64>,
    &'py PyArray2<f32>,
    &'py PyArray1<i64>,
)> {
    let hidden = hidden.as_array();
    let exit_mask = exit_mask.as_array();
    
    let n = hidden.nrows();
    let hidden_size = hidden.ncols();
    
    // 收集索引
    let mut active_indices: Vec<i64> = Vec::new();
    let mut exit_indices: Vec<i64> = Vec::new();
    
    for i in 0..n {
        if exit_mask[i] != 0 {
            exit_indices.push(i as i64);
        } else {
            active_indices.push(i as i64);
        }
    }
    
    let n_active = active_indices.len();
    let n_exit = exit_indices.len();
    
    // 并行收集hidden states
    let active_hidden: Vec<f32> = active_indices
        .par_iter()
        .flat_map(|&i| {
            (0..hidden_size).map(move |j| hidden[[i as usize, j]])
        })
        .collect();
    
    let exit_hidden: Vec<f32> = exit_indices
        .par_iter()
        .flat_map(|&i| {
            (0..hidden_size).map(move |j| hidden[[i as usize, j]])
        })
        .collect();
    
    Ok((
        PyArray2::from_vec(py, active_hidden).unwrap().reshape([n_active, hidden_size]).unwrap(),
        PyArray1::from_vec(py, active_indices),
        PyArray2::from_vec(py, exit_hidden).unwrap().reshape([n_exit, hidden_size]).unwrap(),
        PyArray1::from_vec(py, exit_indices),
    ))
}

/// Token Router Merge (并行)
#[pyfunction]
fn token_router_merge_rust<'py>(
    py: Python<'py>,
    active_hidden: PyReadonlyArray2<f32>,
    active_indices: PyReadonlyArray1<i64>,
    exit_hidden: PyReadonlyArray2<f32>,
    exit_indices: PyReadonlyArray1<i64>,
    total_size: usize,
) -> PyResult<&'py PyArray2<f32>> {
    let active_hidden = active_hidden.as_array();
    let active_indices = active_indices.as_array();
    let exit_hidden = exit_hidden.as_array();
    let exit_indices = exit_indices.as_array();
    
    let hidden_size = if active_hidden.ncols() > 0 {
        active_hidden.ncols()
    } else {
        exit_hidden.ncols()
    };
    
    // 初始化输出
    let mut merged = vec![0.0f32; total_size * hidden_size];
    
    // 并行写入active
    active_indices.iter().enumerate().for_each(|(local_idx, &orig_idx)| {
        let orig_idx = orig_idx as usize;
        for j in 0..hidden_size {
            merged[orig_idx * hidden_size + j] = active_hidden[[local_idx, j]];
        }
    });
    
    // 并行写入exit
    exit_indices.iter().enumerate().for_each(|(local_idx, &orig_idx)| {
        let orig_idx = orig_idx as usize;
        for j in 0..hidden_size {
            merged[orig_idx * hidden_size + j] = exit_hidden[[local_idx, j]];
        }
    });
    
    Ok(PyArray2::from_vec(py, merged).unwrap().reshape([total_size, hidden_size]).unwrap())
}

/// KV投影 (并行GEMM)
#[pyfunction]
fn kv_projection_rust<'py>(
    py: Python<'py>,
    hidden: PyReadonlyArray2<f32>,
    wk: PyReadonlyArray2<f32>,
    wv: PyReadonlyArray2<f32>,
) -> PyResult<(&'py PyArray2<f32>, &'py PyArray2<f32>)> {
    let hidden = hidden.as_array();
    let wk = wk.as_array();
    let wv = wv.as_array();
    
    let n = hidden.nrows();
    let hidden_size = hidden.ncols();
    let kv_dim = wk.ncols();
    
    // 并行矩阵乘法
    let key: Vec<f32> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            (0..kv_dim).map(move |j| {
                (0..hidden_size)
                    .map(|k| hidden[[i, k]] * wk[[k, j]])
                    .sum::<f32>()
            })
        })
        .collect();
    
    let value: Vec<f32> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            (0..kv_dim).map(move |j| {
                (0..hidden_size)
                    .map(|k| hidden[[i, k]] * wv[[k, j]])
                    .sum::<f32>()
            })
        })
        .collect();
    
    Ok((
        PyArray2::from_vec(py, key).unwrap().reshape([n, kv_dim]).unwrap(),
        PyArray2::from_vec(py, value).unwrap().reshape([n, kv_dim]).unwrap(),
    ))
}

/// 在线统计量更新 (Welford算法)
#[pyfunction]
fn welford_update(
    count: i64,
    mean: f32,
    m2: f32,
    new_values: PyReadonlyArray1<f32>,
) -> PyResult<(i64, f32, f32)> {
    let values = new_values.as_array();
    
    let mut count = count;
    let mut mean = mean;
    let mut m2 = m2;
    
    for &x in values.iter() {
        count += 1;
        let delta = x - mean;
        mean += delta / count as f32;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    
    Ok((count, mean, m2))
}

/// Python模块定义
#[pymodule]
fn sedac_kernels(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fused_entropy_decision_rust, m)?)?;
    m.add_function(wrap_pyfunction!(token_router_split_rust, m)?)?;
    m.add_function(wrap_pyfunction!(token_router_merge_rust, m)?)?;
    m.add_function(wrap_pyfunction!(kv_projection_rust, m)?)?;
    m.add_function(wrap_pyfunction!(welford_update, m)?)?;
    Ok(())
}
