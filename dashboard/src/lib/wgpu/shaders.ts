/**
 * Native WGSL shaders for ECG waveform rendering.
 *
 * Architecture: fullscreen-quad fragment shader approach.
 * The fragment shader samples from a signal buffer and renders:
 *   1. Background grid (major + minor lines)
 *   2. Entropy waveform with anti-aliased edges and glow
 *   3. Decision markers as colored dots
 *   4. Threshold lines (FAST/DEEP)
 */

export const ECG_SHADER = /* wgsl */ `

struct Uniforms {
  viewport: vec2<f32>,       // canvas width, height in pixels
  sample_count: f32,         // number of valid samples in buffer
  buffer_size: f32,          // total buffer capacity
  time: f32,                 // elapsed seconds (for animation)
  line_width: f32,           // waveform line width in pixels
  fast_threshold: f32,       // FAST threshold (normalized)
  deep_threshold: f32,       // DEEP threshold (normalized)
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> entropy: array<f32>;
@group(0) @binding(2) var<storage, read> decisions: array<u32>;

// ── Fullscreen triangle (3 vertices cover entire screen) ──
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  return vec4<f32>(pos[vid], 0.0, 1.0);
}

// ── Helpers ──

fn grid_intensity(coord: f32, spacing: f32, width: f32) -> f32 {
  let d = abs(fract(coord / spacing + 0.5) - 0.5) * spacing;
  return smoothstep(width, 0.0, d);
}

fn sample_signal(x_norm: f32) -> f32 {
  let n = u32(u.sample_count);
  if (n == 0u) { return 0.0; }
  let fi = x_norm * f32(n - 1u);
  let i0 = u32(fi);
  let i1 = min(i0 + 1u, n - 1u);
  let t = fi - f32(i0);
  return mix(entropy[i0], entropy[i1], t);
}

fn get_decision(x_norm: f32) -> u32 {
  let n = u32(u.sample_count);
  if (n == 0u) { return 1u; }
  let idx = u32(x_norm * f32(n - 1u) + 0.5);
  return decisions[min(idx, n - 1u)];
}

// Normalize entropy to [0,1] range (entropy domain 0..5)
fn normalize_entropy(e: f32) -> f32 {
  return clamp(e / 5.0, 0.0, 1.0);
}

// Color based on entropy value
fn entropy_color(e_norm: f32) -> vec3<f32> {
  // Green → Yellow → Red gradient
  let green  = vec3<f32>(0.0, 1.0, 0.53);
  let yellow = vec3<f32>(1.0, 0.8, 0.0);
  let red    = vec3<f32>(1.0, 0.17, 0.17);
  if (e_norm < 0.4) {
    return mix(green, yellow, e_norm / 0.4);
  }
  return mix(yellow, red, clamp((e_norm - 0.4) / 0.6, 0.0, 1.0));
}

// ── Fragment shader ──

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = frag.xy / u.viewport;
  let px = frag.xy;

  // Margins: 40px left, 20px right, 20px top/bottom
  let margin = vec4<f32>(40.0, 20.0, 20.0, 20.0); // left, right, top, bottom
  let plot_min = vec2<f32>(margin.x, margin.z);
  let plot_max = u.viewport - vec2<f32>(margin.y, margin.w);
  let plot_size = plot_max - plot_min;

  // Plot-local UV [0,1]
  let puv = (px - plot_min) / plot_size;

  // Background
  var color = vec3<f32>(0.04, 0.06, 0.10);

  // Check if inside plot area
  if (puv.x >= 0.0 && puv.x <= 1.0 && puv.y >= 0.0 && puv.y <= 1.0) {
    // ── Grid ──
    let grid_minor = grid_intensity(puv.x, 0.05, 0.5 / plot_size.x) * 0.08;
    let grid_major = grid_intensity(puv.x, 0.25, 1.0 / plot_size.x) * 0.15;
    let grid_h_minor = grid_intensity(puv.y, 0.1, 0.5 / plot_size.y) * 0.08;
    let grid_h_major = grid_intensity(puv.y, 0.5, 1.0 / plot_size.y) * 0.15;
    let grid = max(max(grid_minor, grid_major), max(grid_h_minor, grid_h_major));
    color += vec3<f32>(0.1, 0.15, 0.25) * grid;

    // Y is flipped: 0 at top, 1 at bottom → flip for signal
    let y_signal = 1.0 - puv.y;

    // ── Threshold lines ──
    let fast_y = normalize_entropy(u.fast_threshold);
    let deep_y = normalize_entropy(u.deep_threshold);

    let fast_dist = abs(y_signal - fast_y) * plot_size.y;
    let deep_dist = abs(y_signal - deep_y) * plot_size.y;

    // Dashed threshold lines
    let dash = step(0.5, fract(puv.x * 40.0));
    color += vec3<f32>(0.0, 0.4, 0.2) * smoothstep(1.5, 0.0, fast_dist) * 0.4 * dash;
    color += vec3<f32>(0.4, 0.1, 0.1) * smoothstep(1.5, 0.0, deep_dist) * 0.4 * dash;

    // ── Waveform ──
    if (u.sample_count > 1.0) {
      let e_val = sample_signal(puv.x);
      let e_norm = normalize_entropy(e_val);
      let signal_y = e_norm;

      let dist = abs(y_signal - signal_y) * plot_size.y;
      let half_w = u.line_width * 0.5;

      // Core line
      let line_alpha = smoothstep(half_w + 1.0, half_w - 0.5, dist);
      let sig_color = entropy_color(e_norm);
      color = mix(color, sig_color, line_alpha);

      // Glow
      let glow_alpha = smoothstep(half_w + 12.0, half_w, dist) * 0.15;
      color += sig_color * glow_alpha;

      // ── Decision markers ──
      let dec = get_decision(puv.x);
      if (dist < 6.0) {
        if (dec == 0u) {
          // FAST: green dot
          let dot = smoothstep(5.0, 3.0, dist);
          color = mix(color, vec3<f32>(0.0, 1.0, 0.53), dot * 0.5);
        } else if (dec == 2u) {
          // DEEP: red dot with ring
          let dot = smoothstep(6.0, 4.0, dist);
          color = mix(color, vec3<f32>(1.0, 0.2, 0.2), dot * 0.6);
        }
      }
    }
  }

  // Border highlight on plot edges
  let border_l = smoothstep(1.5, 0.0, abs(px.x - plot_min.x));
  let border_r = smoothstep(1.5, 0.0, abs(px.x - plot_max.x));
  let border_t = smoothstep(1.5, 0.0, abs(px.y - plot_min.y));
  let border_b = smoothstep(1.5, 0.0, abs(px.y - plot_max.y));
  let border = max(max(border_l, border_r), max(border_t, border_b));
  color += vec3<f32>(0.15, 0.25, 0.4) * border * 0.5;

  return vec4<f32>(color, 1.0);
}
`;
