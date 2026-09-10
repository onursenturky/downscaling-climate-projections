"""Daily ERA5 -> ERA5-Land precipitation experiment for Istanbul.

All preprocessing is explicit about UTC dates, spatial grids and missing data.
This module never downloads data or starts training on import.
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import xarray as xr


PRED_VARS = [f"{v}_{p}" for v in ("u", "v", "q", "t", "z") for p in (850, 700, 500)]


def read_netcdf(path):
    """Load NetCDF or CDS ZIP content, including archives with multiple variables."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {path}")
    if not zipfile.is_zipfile(path):
        with xr.open_dataset(path) as ds:
            return ds.load()
    parts = []
    with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory() as directory:
        members = [name for name in archive.namelist() if name.endswith(".nc")]
        if not members:
            raise ValueError(f"Arşivde NetCDF yok: {path}")
        for i, member in enumerate(members):
            # Use our own filename rather than extracting arbitrary archive paths.
            target = Path(directory) / f"part_{i}.nc"
            target.write_bytes(archive.read(member))
            with xr.open_dataset(target) as ds:
                parts.append(ds.load())
    return xr.merge(parts, join="exact", compat="no_conflicts")


def canonical_coordinates(ds):
    names = {old: new for old, new in (
        ("valid_time", "time"), ("latitude", "lat"),
        ("longitude", "lon"), ("level", "pressure_level"),
    ) if old in ds.dims}
    ds = ds.rename(names)
    ds = ds.drop_vars([v for v in ("number", "expver") if v in ds], errors="ignore")
    for dim in ("time", "lat", "lon"):
        if dim not in ds.dims:
            raise ValueError(f"Eksik boyut: {dim}")
        if not ds.indexes[dim].is_unique:
            raise ValueError(f"Tekrarlanan koordinat: {dim}")
    if not np.issubdtype(ds.time.dtype, np.datetime64):
        raise ValueError("Bu ERA5 akışı Gregorian UTC datetime64 tarihleri bekler.")
    ds = ds.sortby(["time", "lat", "lon"])
    for dim in ("lat", "lon"):
        if "stored_direction" in ds[dim].attrs:
            ds[dim].attrs["stored_direction"] = "increasing"
    return ds


def require_same_grid(actual, expected, context="Veri"):
    for dim in ("lat", "lon"):
        if not actual.indexes[dim].equals(expected.indexes[dim]):
            raise ValueError(f"{context}: {dim} koordinatları veya sırası farklı.")


def load_months(paths):
    paths = sorted(map(Path, paths))
    if not paths:
        raise FileNotFoundError("Aylık ham NetCDF dosyaları bulunamadı.")
    parts = [canonical_coordinates(read_netcdf(path)) for path in paths]
    for part in parts[1:]:
        require_same_grid(part, parts[0], "Aylık dosyalar")
    ds = xr.concat(parts, dim="time", data_vars="minimal", coords="minimal",
                   compat="equals", join="exact").sortby("time")
    if not ds.indexes["time"].is_unique:
        raise ValueError("Aylık dosyalarda tekrarlanan zamanlar var.")
    return ds


def require_sampling(ds, hours):
    """Require a complete, continuous raw interval; never average partial days."""
    start = pd.Timestamp(ds.time.values[0]).normalize()
    end = pd.Timestamp(ds.time.values[-1]).normalize()
    expected = pd.DatetimeIndex([
        day + pd.Timedelta(hours=hour)
        for day in pd.date_range(start, end, freq="D") for hour in hours
    ])
    if not ds.indexes["time"].equals(expected):
        missing = expected.difference(ds.indexes["time"])
        raise ValueError(f"Ham zaman örneklemesi eksik/uyumsuz. Eksik örnekler: {list(missing[:5])}")


def prepare_daily(predictor_paths, target_paths):
    raw_x, raw_y = load_months(predictor_paths), load_months(target_paths)
    require_sampling(raw_x, [0, 6, 12, 18])
    require_sampling(raw_y, list(range(24)))
    expected_units = {"u": "m s**-1", "v": "m s**-1", "q": "kg kg**-1",
                      "t": "K", "z": "m**2 s**-2"}
    for name, unit in expected_units.items():
        if name not in raw_x or raw_x[name].attrs.get("units") != unit:
            raise ValueError(f"ERA5 {name}: beklenen birim {unit}.")
    if raw_y.tp.attrs.get("units") != "m" or raw_y.t2m.attrs.get("units") != "K":
        raise ValueError("ERA5-Land tp metre, t2m Kelvin olmalıdır.")
    if raw_y.tp.attrs.get("GRIB_stepType") != "accum":
        raise ValueError("Bu akış ERA5-Land birikimli tp verisini bekler.")
    if raw_x.pressure_level.attrs.get("units") != "hPa":
        raise ValueError("Basınç seviyeleri hPa olmalıdır.")
    if not np.isfinite(raw_x[list(expected_units)].to_array().values).all():
        raise ValueError("Ham ERA5 girdisinde eksik/sonlu olmayan değer var.")
    for name in ("tp", "t2m"):
        valid = np.isfinite(raw_y[name].transpose("time", "lat", "lon").values)
        if not np.array_equal(valid, np.broadcast_to(valid[0], valid.shape)):
            raise ValueError(f"Ham ERA5-Land {name}: maske değişiyor veya saatlik veri eksik.")
    daily_x = raw_x.resample(time="1D").mean(keep_attrs=True)
    fields = {}
    for name in PRED_VARS:
        variable, pressure = name.split("_")
        fields[name] = daily_x[variable].sel(pressure_level=int(pressure), drop=True)
    predictors = xr.Dataset(fields)[PRED_VARS].transpose("time", "lat", "lon")
    # The 00 UTC value is the preceding day's total, NOT an hourly increment.
    pr = raw_y.tp.sel(time=raw_y.time.dt.hour == 0) * 1000.0
    pr = pr.assign_coords(time=pr.time - np.timedelta64(1, "D"))
    pr.attrs = {"units": "mm/day", "long_name": "Daily total precipitation (UTC)"}
    tasmax = raw_y.t2m.resample(time="1D").max() - 273.15
    tasmax.attrs = {"units": "degC", "long_name": "Maximum hourly 2m temperature per UTC day"}
    common = predictors.indexes["time"].intersection(pr.indexes["time"]).intersection(tasmax.indexes["time"])
    if common.empty:
        raise ValueError("Girdi ve hedefin ortak günü yok.")
    excluded = predictors.indexes["time"].difference(common)
    internal = excluded[(excluded > common.min()) & (excluded < common.max())]
    if len(internal):
        raise ValueError(f"Dönem içinde kayıp hedef günleri: {list(internal)}")
    predictors = predictors.sel(time=common)
    targets = xr.Dataset({"pr": pr.sel(time=common), "tasmax": tasmax.sel(time=common)})
    for ds in (predictors, targets):
        ds.attrs.update(time_basis="UTC", purpose="Istanbul reanalysis experiment; not a climate projection")
    audit = {
        "days": len(common), "start": str(common[0].date()), "end": str(common[-1].date()),
        "excluded_predictor_dates": [str(t.date()) for t in excluded],
        "predictor_files": [str(p) for p in predictor_paths],
        "target_files": [str(p) for p in target_paths],
    }
    return predictors, targets, audit


def align_time(pred, obs):
    """Validate exact date identity; never relabel predictions to hide an offset."""
    if not pred.indexes["time"].equals(obs.indexes["time"]):
        raise ValueError("Tahmin ve referans tarihleri/sırası aynı değil.")
    require_same_grid(pred, obs, "Tahmin/referans")
    return pred


def split_periods(predictors, precip, periods):
    if not predictors.indexes["time"].equals(precip.indexes["time"]):
        raise ValueError("Girdi/hedef günleri farklı.")
    result = {}
    previous_end = None
    for name in ("train", "validation", "test"):
        start, end = map(pd.Timestamp, periods[name])
        if end < start or (previous_end is not None and start <= previous_end):
            raise ValueError("Eğitim, doğrulama ve test tarihleri sıralı ve ayrık olmalıdır.")
        expected = pd.date_range(start, end, freq="D")
        missing = expected.difference(predictors.indexes["time"])
        if len(missing):
            raise ValueError(f"{name}: eksik günler: {list(missing[:5])}")
        result[name] = (predictors.sel(time=expected), precip.sel(time=expected))
        previous_end = end
    return result


@dataclass
class Preprocessor:
    mean: xr.Dataset
    std: xr.Dataset
    land_mask: xr.DataArray
    variables: list

    def inputs(self, predictors):
        require_same_grid(predictors, self.mean, "Model girdisi")
        if not predictors.indexes["time"].is_unique or not predictors.indexes["time"].is_monotonic_increasing:
            raise ValueError("Girdi tarihleri benzersiz ve artan olmalıdır.")
        for name in self.variables:
            if name not in predictors:
                raise ValueError(f"Eksik predictor: {name}")
            if predictors[name].attrs.get("units") != self.mean[name].attrs.get("units"):
                raise ValueError(f"{name}: eğitim ve tahmin birimleri farklı.")
        standardized = (predictors[self.variables] - self.mean) / self.std
        arr = standardized.to_array().transpose("time", "variable", "lat", "lon").values.astype("float32")
        if not len(arr) or not np.isfinite(arr).all():
            raise ValueError("Girdi boş veya sonlu olmayan değer içeriyor.")
        return arr

    def targets(self, precip):
        require_same_grid(precip, self.land_mask, "Hedef")
        values = precip.transpose("time", "lat", "lon").values
        expected = np.broadcast_to(self.land_mask.values, values.shape)
        if not np.array_equal(np.isfinite(values), expected):
            raise ValueError("Hedefin geçerli hücre maskesi değişmiş veya karada eksik veri var.")
        arr = values[:, self.land_mask.values].astype("float32")
        if not len(arr) or (arr < 0).any():
            raise ValueError("Hedef boş veya negatif yağış içeriyor.")
        return arr

    def to_grid(self, values, time):
        expected_shape = (len(time), int(self.land_mask.sum()))
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"Geçersiz model çıktısı; beklenen boyut {expected_shape}.")
        grid = np.full((len(time), *self.land_mask.shape), np.nan, dtype="float32")
        grid[:, self.land_mask.values] = values
        return xr.DataArray(grid, dims=("time", "lat", "lon"),
                            coords={"time": time, "lat": self.land_mask.lat, "lon": self.land_mask.lon},
                            name="pr", attrs={"units": "mm/day"})


def fit_preprocessor(train_x, train_y):
    if not train_x.indexes["time"].equals(train_y.indexes["time"]):
        raise ValueError("Eğitim girdisi ve hedef tarihleri farklı.")
    if train_x.sizes["time"] < 2:
        raise ValueError("Normalizasyon için en az iki eğitim günü gerekir.")
    mean = train_x[PRED_VARS].mean("time", keep_attrs=True)
    std = train_x[PRED_VARS].std("time", keep_attrs=True)
    for name in PRED_VARS:
        std[name] = std[name].where(std[name] > 1e-8, 1.0)
    mask = train_y.isel(time=0, drop=True).notnull().transpose("lat", "lon")
    if not mask.any():
        raise ValueError("Geçerli hedef hücresi yok.")
    prep = Preprocessor(mean, std, mask, list(PRED_VARS))
    prep.inputs(train_x)
    prep.targets(train_y)
    return prep


class DeepESD(nn.Module):
    """Three convolutions and a dense, nonnegative precipitation output."""

    def __init__(self, n_predictors, n_outputs, in_hw, filters_last_conv=4):
        super().__init__()
        self.config = dict(n_predictors=n_predictors, n_outputs=n_outputs,
                           in_hw=list(in_hw), filters_last_conv=filters_last_conv)
        self.features = nn.Sequential(
            nn.Conv2d(n_predictors, 50, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(50, 25, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(25, filters_last_conv, 3, padding=1), nn.LeakyReLU(0.1),
        )
        self.out = nn.Linear(int(np.prod(in_hw)) * filters_last_conv, n_outputs)

    def forward(self, x):
        return F.softplus(self.out(self.features(x).flatten(1)))

    def initialize_from_mean(self, training_mean):
        """Start near the training-only baseline instead of almost-zero rain."""
        mean = torch.as_tensor(training_mean, dtype=torch.float32).clamp(min=1e-4)
        with torch.no_grad():
            self.out.weight.normal_(mean=0.0, std=1e-3)
            self.out.bias.copy_(mean + torch.log(-torch.expm1(-mean)))


def train_model(model, train_x, train_y, val_x, val_y, *, epochs=200,
                batch_size=16, lr=1e-3, patience=25, seed=0, device="cpu"):
    """Select weights by validation loss only. Test data is not an argument."""
    if min(len(train_x), len(val_x)) == 0:
        raise ValueError("Eğitim/doğrulama boş olamaz.")
    model = model.to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                        batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    vx, vy = torch.from_numpy(val_x).to(device), torch.from_numpy(val_y).to(device)
    with torch.no_grad():
        model.eval()
        initial_val = float(F.mse_loss(model(vx), vy))
    best_loss, best_epoch, stale = initial_val, 0, 0
    best_state = deepcopy(model.state_dict())
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Eğitim kaybı sonlu değil.")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
        model.eval()
        with torch.no_grad():
            val_loss = float(F.mse_loss(model(vx), vy))
        if not np.isfinite(val_loss):
            raise RuntimeError("Doğrulama kaybı sonlu değil.")
        history.append({"epoch": epoch, "train_mse": total / len(train_x), "validation_mse": val_loss})
        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"Epoch {epoch:3d}: train MSE={total / len(train_x):.3f}, validation MSE={val_loss:.3f}")
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    info = {"best_epoch": best_epoch, "best_validation_mse": best_loss,
            "initial_validation_mse": initial_val, "epochs_run": len(history)}
    return pd.DataFrame(history), info


def downscale(model, predictors, prep, batch_size=256):
    inputs = prep.inputs(predictors)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        values = np.concatenate([
            model(torch.from_numpy(inputs[i:i + batch_size]).to(device)).cpu().numpy()
            for i in range(0, len(inputs), batch_size)
        ])
    return prep.to_grid(values, predictors.time)


def baseline_predictions(training_target, target_time, prep):
    values = prep.targets(training_target)
    shape = (len(target_time), values.shape[1])
    return {
        "Sıfır yağış": prep.to_grid(np.zeros(shape, dtype="float32"), target_time),
        "Eğitim genel ortalaması": prep.to_grid(np.full(shape, values.mean(), dtype="float32"), target_time),
        "Eğitim hücre ortalaması": prep.to_grid(np.broadcast_to(values.mean(0), shape), target_time),
    }


def summarize(obs, pred, prep, wet_threshold=1.0):
    align_time(pred, obs)
    actual, estimate = prep.targets(obs), prep.targets(pred)
    wet_actual, wet_estimate = actual >= wet_threshold, estimate >= wet_threshold
    return {
        "RMSE_mm_day": float(np.sqrt(np.mean((estimate - actual) ** 2))),
        "MAE_mm_day": float(np.abs(estimate - actual).mean()),
        "bias_mm_day": float((estimate - actual).mean()),
        "mean_prediction_mm_day": float(estimate.mean()),
        "mean_reference_mm_day": float(actual.mean()),
        "wet_prediction_percent": float(wet_estimate.mean() * 100),
        "wet_reference_percent": float(wet_actual.mean() * 100),
        "SDII_prediction_mm_day": float(estimate[wet_estimate].mean()) if wet_estimate.any() else np.nan,
        "SDII_reference_mm_day": float(actual[wet_actual].mean()) if wet_actual.any() else np.nan,
    }


def save_checkpoint(path, model, prep, metadata):
    """Store tensors and primitive metadata, readable with weights_only=True."""
    bundle = {
        "format_version": 1, "model_config": model.config,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "variables": prep.variables,
        "units": {name: prep.mean[name].attrs.get("units", "") for name in prep.variables},
        "predictor_lat": torch.tensor(prep.mean.lat.values),
        "predictor_lon": torch.tensor(prep.mean.lon.values),
        "mean": torch.tensor(prep.mean[prep.variables].to_array().values),
        "std": torch.tensor(prep.std[prep.variables].to_array().values),
        "target_lat": torch.tensor(prep.land_mask.lat.values),
        "target_lon": torch.tensor(prep.land_mask.lon.values),
        "land_mask": torch.tensor(prep.land_mask.values),
        "coordinate_attrs": {
            "predictor": {dim: dict(prep.mean[dim].attrs) for dim in ("lat", "lon")},
            "target": {dim: dict(prep.land_mask[dim].attrs) for dim in ("lat", "lon")},
        },
        "metadata": metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(bundle, temp)
    temp.replace(path)


def load_checkpoint(path):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format_version") != 1:
        raise ValueError("Desteklenmeyen model dosyası; yalnız ağırlık içeren eski dosya kullanılamaz.")
    coords = {"lat": bundle["predictor_lat"].numpy(), "lon": bundle["predictor_lon"].numpy()}
    def dataset(key):
        return xr.Dataset({name: xr.DataArray(bundle[key][i].numpy(), dims=("lat", "lon"),
                                              coords=coords, attrs={"units": bundle["units"][name]})
                           for i, name in enumerate(bundle["variables"])})
    mask = xr.DataArray(bundle["land_mask"].numpy(), dims=("lat", "lon"),
                        coords={"lat": bundle["target_lat"].numpy(), "lon": bundle["target_lon"].numpy()})
    mean, std = dataset("mean"), dataset("std")
    for dim in ("lat", "lon"):
        mean[dim].attrs = bundle["coordinate_attrs"]["predictor"][dim]
        std[dim].attrs = bundle["coordinate_attrs"]["predictor"][dim]
        mask[dim].attrs = bundle["coordinate_attrs"]["target"][dim]
    prep = Preprocessor(mean, std, mask, bundle["variables"])
    model = DeepESD(**bundle["model_config"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, prep, bundle["metadata"]
