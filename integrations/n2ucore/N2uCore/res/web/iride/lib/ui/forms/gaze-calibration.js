import { Texts } from "../../core/texts.js";
import { validPoint } from "./gaze-ui.js";

export function calibrationTargets(width, height) {
  const columns = 4;
  return Array.from({ length: 16 }, (_, i) => [
    Math.round(width * (.08 + .84 * (i % columns) / (columns - 1))),
    Math.round(height * [.08, .37, .63, .92][Math.floor(i / columns)])
  ]);
}

export function validationTargets(width, height) {
  return [.20, .50, .80].flatMap(y => [.18, .50, .82].map(x => [Math.round(x * width), Math.round(y * height)]));
}

function percentile(values, fraction) {
  const sorted = [...values].sort((a, b) => a - b), index = (sorted.length - 1) * fraction;
  const low = Math.floor(index), high = Math.ceil(index);
  return sorted[low] + (sorted[high] - sorted[low]) * (index - low);
}

export class GazeValidation {
  #size; #active = null; #targets = [];
  constructor(size) { this.#size = size; }
  beginTarget(target, now, sourceAfterUs) {
    if (!Number.isSafeInteger(sourceAfterUs)) throw new Error(Texts.get("gaze.validationTimestampMissing"));
    this.#active = { target, started: now, sourceAfterUs, latest: sourceAfterUs, errors: [] };
  }
  observe(sample, now) {
    const active = this.#active, ts = sample?.source_ts_us;
    if (!active || !validPoint(sample, this.#size) || !Number.isSafeInteger(ts) || ts <= active.latest) return;
    active.latest = ts;
    if (now - active.started < 1000 || now - active.started > 3000 || ts < active.sourceAfterUs + 1000000 || active.errors.length >= 128) return;
    const [x, y] = sample.screen_xy_px;
    active.errors.push(Math.hypot(x - active.target[0], y - active.target[1]));
  }
  finishTarget() {
    const active = this.#active;
    if (!active || active.errors.length < 3) throw new Error(Texts.get("gaze.validationSamplesMissing"));
    this.#targets.push(active); this.#active = null;
  }
  result(calibrationPoints) {
    if (this.#active || this.#targets.length !== 9) throw new Error(Texts.get("gaze.validationTargetsMissing"));
    const targets = this.#targets.map(({ target, errors }) => ({
      target_px: target, samples: errors.length,
      mean_error_px: errors.reduce((a, b) => a + b, 0) / errors.length,
      mean_squared_error_px: errors.reduce((a, b) => a + b * b, 0) / errors.length
    }));
    const errors = this.#targets.flatMap(target => target.errors);
    return {
      schema: "screen-gaze-validation.v1", calibration_points: calibrationPoints,
      coordinate_space: "viewport_css_pixels", viewport: { ...this.#size },
      tested_at: new Date().toISOString(), target_count: targets.length,
      samples: errors.length, mean_error_px: targets.reduce((sum, target) => sum + target.mean_error_px, 0) / targets.length,
      rmse_px: Math.sqrt(targets.reduce((sum, target) => sum + target.mean_squared_error_px, 0) / targets.length),
      median_error_px: percentile(errors, .5), p95_error_px: percentile(errors, .95), targets
    };
  }
}
