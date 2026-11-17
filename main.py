#!/usr/bin/env python3
"""Forecast a launch day's hourly Twitch viewers, hour by hour.

The program reads one case file (schema makefun/twitch-ccu-case/v1) from
the input directory and writes one forecast file per case to the output
directory.  Each forecast gives five quantiles (5/25/50/75/95%) of the
category's concurrent viewers for hours 2..24 of the release day.

How it forecasts
---------------
The only anchor the case gives is the first hour's print, p1.  Across
past big launches the later hours relate to p1 in a strongly regular way
once everything is taken in logs: a launch that is already huge in hour 1
(its audience unlocked everywhere at once) barely grows afterwards, while
a launch that is still small in hour 1 (only a few regions unlocked, the
rest still to come) grows several-fold by the evening.  So for each hour
h the program regresses log viewers in hour h on log viewers in hour 1
over the launch curves bundled with the case, using both a straight line
and a quadratic (the quadratic lets the growth saturate for the very
biggest launches), and blends the two medians.  It then blends that
regression median with a second one: the median curve of the few past
launches whose first hour was closest in size to this case's, which
catches local structure the smooth fits miss.  The spread of the
quantiles comes from the scatter of past launches around those fits, so
hours where launches disagree get wide intervals and hours where they
agree get narrow ones.  Everything is computed from the case file at run
time; no number from any launch is baked in.

The program is deterministic, uses only the standard library, and never
crashes on odd input: any failure in the model falls back to a plain
persistence forecast, and any failure reading a case still writes a
validly shaped output for it.
"""

import json
import math
import os
from datetime import datetime, timezone

# Standard normal quantiles at the five levels the forecast reports.
NORMAL_Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Weight of the quadratic fit in the blended median (the rest is linear).
QUAD_WEIGHT = 0.75
# A launch curve is usable only if its release day is mostly present.
MIN_DAY_HOURS = 20
# Least-squares needs at least this many launches for one hour's fit.
MIN_FIT_POINTS = 10
# Weight of the neighbour median in the final blended median.
NEIGHBOUR_WEIGHT = 0.5
# How many past launches, nearest by hour-1 size, form the neighbourhood.
NEIGHBOUR_COUNT = 10
# A neighbourhood needs at least this many launches with the hour.
NEIGHBOUR_MIN_POINTS = 5
# Floor and ceiling on the log-scale spread of an hour's quantiles.
SIGMA_FLOOR = 0.10
SIGMA_CEILING = 1.50
# The baseline spread growth used when no fit is possible at all.
BASELINE_SIGMA_RATE = 0.15
# The quadratic feature is scaled so the normal equations stay tame.
QUAD_SCALE = 10.0
VIEWER_FLOOR = 1.0
VIEWER_CEILING = 1e9


def parse_time(text):
    """Parse an ISO 8601 timestamp (with a Z suffix) to an aware datetime."""
    if not isinstance(text, str):
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def day_hours(comp):
    """Hourly viewers of one comparable's release day, keyed by hour 1..24.

    Returns an empty dict when the comparable has no usable release day
    (for instance the target game's own pre-release history, which ends
    before its release day begins).
    """
    start = parse_time(comp.get("day_start"))
    if start is None:
        return {}
    hours = {}
    for point in comp.get("hourly") or []:
        moment = parse_time(point.get("from"))
        if moment is None:
            continue
        offset = int((moment - start).total_seconds() // 3600) + 1
        if 1 <= offset <= 24:
            viewers = point.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                hours[offset] = float(viewers)
    return hours


def solve_linear(matrix, rhs):
    """Solve a small dense linear system by Gaussian elimination.

    Returns the solution list, or None when the system is singular.
    """
    size = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for other in range(col + 1, size):
            factor = rows[other][col] / rows[col][col]
            for k in range(col, size + 1):
                rows[other][k] -= factor * rows[col][k]
    out = [0.0] * size
    for i in range(size - 1, -1, -1):
        total = rows[i][size]
        for j in range(i + 1, size):
            total -= rows[i][j] * out[j]
        out[i] = total / rows[i][i]
    return out


def least_squares(features, targets):
    """Ordinary least squares for targets ~ features (with intercept).

    `features` is a list of per-point feature lists (without the
    intercept, which is added here).  Returns (coefficients, residual
    standard deviation) or None when the fit cannot be made.
    """
    count = len(targets)
    width = len(features[0]) + 1 if features else 0
    if count < width + 1:
        return None
    design = [[1.0] + list(row) for row in features]
    normal = [[sum(design[k][i] * design[k][j] for k in range(count))
               for j in range(width)] for i in range(width)]
    moments = [sum(design[k][i] * targets[k] for k in range(count))
               for i in range(width)]
    coeffs = solve_linear(normal, moments)
    if coeffs is None:
        return None
    squared = 0.0
    for row, target in zip(design, targets):
        fitted = sum(c * x for c, x in zip(coeffs, row))
        squared += (target - fitted) ** 2
    dof = max(count - width, 1)
    return coeffs, math.sqrt(squared / dof)


def features_for(log_first, quadratic):
    """Feature list (no intercept) for a launch given its log hour-1."""
    if quadratic:
        return [log_first, log_first * log_first / QUAD_SCALE]
    return [log_first]


def fit_hour(curves, hour):
    """Fit log viewers at `hour` against log hour-1 over past launches.

    Returns (median_log, sigma) for a caller-supplied log hour-1 through
    a small closure, or None when no fit is possible.  The median blends
    a linear and a quadratic fit; the sigma averages their residual
    spreads.
    """
    pairs = []
    for hours in curves:
        if hour in hours and 1 in hours:
            pairs.append((math.log(hours[1]), math.log(hours[hour])))
    if len(pairs) < MIN_FIT_POINTS:
        return None
    fits = []
    for quadratic in (False, True):
        feats = [features_for(x, quadratic) for x, _ in pairs]
        targets = [y for _, y in pairs]
        result = least_squares(feats, targets)
        if result is not None:
            fits.append(result)
    if not fits:
        return None

    def predict(log_first):
        medians = []
        sigmas = []
        for coeffs, sigma in fits:
            quad = len(coeffs) == 3
            row = [1.0] + features_for(log_first, quad)
            medians.append(sum(c * x for c, x in zip(coeffs, row)))
            sigmas.append(sigma)
        if len(medians) == 2:
            median = (QUAD_WEIGHT * medians[1]
                      + (1.0 - QUAD_WEIGHT) * medians[0])
        else:
            median = medians[0]
        sigma = sum(sigmas) / len(sigmas)
        sigma = max(SIGMA_FLOOR, min(SIGMA_CEILING, sigma))
        return median, sigma

    return predict


def neighbour_forecast(curves, hour, log_first):
    """Median and spread from the launches nearest in hour-1 size.

    Past launches whose first hour was about the same size as this case's
    are the most informative about how the day unfolds, so this looks at
    the closest few (in log hour-1) and takes the median log ratio to
    hour 1 among them, with the spread from their interquartile range.
    Returns (centre_log, sigma) or None when too few neighbours have
    the hour.
    """
    ranked = sorted(
        curves,
        key=lambda hours: abs(math.log(hours[1]) - log_first),
    )
    near = ranked[:NEIGHBOUR_COUNT]
    ratios = sorted(
        math.log(hours[hour] / hours[1])
        for hours in near
        if hour in hours and 1 in hours
    )
    if len(ratios) < NEIGHBOUR_MIN_POINTS:
        return None
    count = len(ratios)
    median = ratios[count // 2]
    lower = ratios[count // 4]
    upper = ratios[(3 * count) // 4]
    sigma = max(SIGMA_FLOOR, (upper - lower) / 1.349)
    sigma = min(SIGMA_CEILING, sigma)
    return log_first + median, sigma


def median_ratio_forecast(curves, hour, first_print):
    """Fallback: median log ratio to hour 1 over past launches.

    Used when too few launches have hour `hour` for a regression.
    """
    ratios = []
    for hours in curves:
        if hour in hours and 1 in hours:
            ratios.append(math.log(hours[hour] / hours[1]))
    if len(ratios) < 3:
        return None
    ratios.sort()
    count = len(ratios)
    median = ratios[count // 2]
    lower = ratios[count // 4]
    upper = ratios[(3 * count) // 4]
    sigma = max(SIGMA_FLOOR, (upper - lower) / 1.349)
    sigma = min(SIGMA_CEILING, sigma)
    centre = math.log(max(first_print, VIEWER_FLOOR)) + median
    return centre, sigma


def quantiles_from(centre_log, sigma):
    """Five quantiles from a log-median and a log spread, kept valid."""
    values = []
    previous = VIEWER_FLOOR
    for z in NORMAL_Z:
        value = math.exp(centre_log + z * sigma)
        if not math.isfinite(value):
            value = VIEWER_CEILING
        value = max(VIEWER_FLOOR, min(VIEWER_CEILING, value))
        value = round(max(value, previous), 2)
        values.append(value)
        previous = value
    return values


def persistence_row(first_print, hour):
    """Last-resort forecast: stay at hour 1, spread widening over the day."""
    sigma = BASELINE_SIGMA_RATE * math.sqrt(hour - 1)
    return quantiles_from(math.log(max(first_print, VIEWER_FLOOR)), sigma)


def forecast_case(case):
    """Build the 23 forecast rows for one parsed case object."""
    known = case.get("known") or []
    first_print = VIEWER_FLOOR
    if known and isinstance(known[0], dict):
        raw = known[0].get("viewers")
        if isinstance(raw, (int, float)) and raw > 0:
            first_print = float(raw)

    curves = []
    for comp in case.get("comparables") or []:
        if not isinstance(comp, dict):
            continue
        hours = day_hours(comp)
        if hours.get(1) and len(hours) >= MIN_DAY_HOURS:
            curves.append(hours)

    log_first = math.log(max(first_print, VIEWER_FLOOR))
    rows = []
    for hour in range(2, 25):
        fitted = fit_hour(curves, hour)
        if fitted is not None:
            centre, sigma = fitted(log_first)
        else:
            centre = sigma = None
        near = neighbour_forecast(curves, hour, log_first)
        if near is not None:
            if centre is None:
                centre, sigma = near
            else:
                centre = ((1.0 - NEIGHBOUR_WEIGHT) * centre
                          + NEIGHBOUR_WEIGHT * near[0])
                sigma = ((1.0 - NEIGHBOUR_WEIGHT) * sigma
                         + NEIGHBOUR_WEIGHT * near[1])
        if centre is None:
            fallback = median_ratio_forecast(curves, hour, first_print)
            if fallback is not None:
                centre, sigma = fallback
            else:
                rows.append({"hour": hour,
                             "q": persistence_row(first_print, hour)})
                continue
        rows.append({"hour": hour,
                     "q": quantiles_from(centre, sigma)})
    return rows


def read_case(path):
    """Read one case file; returns None when it cannot be parsed."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def write_output(path, rows):
    """Write one forecast file, creating its directory if needed."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = json.dumps({"hours": rows}, separators=(",", ":"))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(payload)


def run():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    names = sorted(
        name for name in os.listdir(input_dir)
        if name.endswith(".in") and os.path.isfile(
            os.path.join(input_dir, name))
    )
    for name in names:
        stem = name[:-3]
        out_path = os.path.join(output_dir, stem + ".out")
        case = read_case(os.path.join(input_dir, name))
        try:
            if case is None:
                raise ValueError("unreadable case")
            rows = forecast_case(case)
        except Exception:
            rows = [{"hour": hour, "q": persistence_row(VIEWER_FLOOR, hour)}
                    for hour in range(2, 25)]
        write_output(out_path, rows)


if __name__ == "__main__":
    run()
