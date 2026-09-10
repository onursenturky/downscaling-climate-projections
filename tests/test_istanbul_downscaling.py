"""Offline regression checks for date, grid, split and inference correctness."""

from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd
import torch
import xarray as xr

from istanbul_downscaling import (
    PRED_VARS, DeepESD, align_time, baseline_predictions, downscale,
    fit_preprocessor, load_checkpoint, prepare_daily, read_netcdf,
    save_checkpoint, split_periods, summarize, train_model,
)


def example():
    rng = np.random.default_rng(3)
    time = pd.date_range("1981-01-01", periods=10)
    x = xr.Dataset({v: xr.DataArray(rng.normal(size=(10, 3, 4)).astype("float32"),
                                   dims=("time", "lat", "lon"),
                                   coords={"time": time, "lat": [40., 41., 42.], "lon": [27., 28., 29., 30.]},
                                   attrs={"units": "test_unit"}) for v in PRED_VARS})
    values = rng.uniform(0, 10, size=(10, 2, 3)).astype("float32")
    values[:, 0, 0] = np.nan
    y = xr.DataArray(values, dims=("time", "lat", "lon"),
                     coords={"time": time, "lat": [40.5, 41.5], "lon": [28., 29., 30.]})
    for ds in (x, y):
        ds.lat.attrs = {"units": "degrees_north", "standard_name": "latitude"}
        ds.lon.attrs = {"units": "degrees_east", "standard_name": "longitude"}
    return x, y


class InferenceContractTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y = example()
        self.prep = fit_preprocessor(self.x.isel(time=slice(0, 6)), self.y.isel(time=slice(0, 6)))

    def test_shifted_dates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "tarih"):
            align_time(self.y.assign_coords(time=self.y.time + np.timedelta64(10, "D")), self.y)

    def test_same_shape_different_grid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "koordinat"):
            self.prep.inputs(self.x.assign_coords(lon=self.x.lon + .1))

    def test_variable_order_is_restored(self):
        np.testing.assert_array_equal(self.prep.inputs(self.x), self.prep.inputs(self.x[list(reversed(PRED_VARS))]))

    def test_unit_mismatch_is_rejected(self):
        changed = self.x.copy(deep=True)
        changed["z_500"].attrs["units"] = "wrong"
        with self.assertRaisesRegex(ValueError, "birim"):
            self.prep.inputs(changed)

    def test_validation_values_do_not_refit_normalization(self):
        shifted = self.x.copy(deep=True)
        for name in PRED_VARS:
            shifted[name].values[6:] += 1000
        before = self.prep.mean.copy(deep=True)
        values = self.prep.inputs(shifted)
        xr.testing.assert_identical(before, self.prep.mean)
        self.assertGreater(float(values[6:].mean()), 100)

    def test_mask_round_trip_and_missing_land_value(self):
        grid = self.prep.to_grid(self.prep.targets(self.y), self.y.time)
        np.testing.assert_array_equal(grid.values, self.y.values)
        broken = self.y.copy(deep=True)
        broken.values[-1, 1, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "eksik"):
            self.prep.targets(broken)

    def test_empty_and_nonfinite_predictors_are_rejected(self):
        with self.assertRaises(ValueError):
            self.prep.inputs(self.x.isel(time=slice(0, 0)))
        broken = self.x.copy(deep=True)
        broken.u_850.values[0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            self.prep.inputs(broken)

    def test_overlapping_periods_are_rejected(self):
        periods = {"train": ["1981-01-01", "1981-01-06"],
                   "validation": ["1981-01-06", "1981-01-08"], "test": ["1981-01-09", "1981-01-10"]}
        with self.assertRaisesRegex(ValueError, "ayrık"):
            split_periods(self.x, self.y, periods)

    def test_requested_missing_day_is_rejected(self):
        periods = {"train": ["1981-01-01", "1981-01-06"],
                   "validation": ["1981-01-07", "1981-01-08"], "test": ["1981-01-09", "1981-01-11"]}
        with self.assertRaisesRegex(ValueError, "eksik gün"):
            split_periods(self.x, self.y, periods)

    def test_baseline_uses_only_provided_training_days(self):
        predictions = baseline_predictions(self.y.isel(time=slice(0, 6)), self.y.time[6:], self.prep)
        expected = self.prep.targets(self.y)[:6].mean(0)
        np.testing.assert_array_equal(self.prep.targets(predictions["Eğitim hücre ortalaması"])[0], expected)
        summary = summarize(self.y.isel(time=slice(6, None)), predictions["Sıfır yağış"], self.prep)
        self.assertTrue(np.isnan(summary["SDII_prediction_mm_day"]))
        self.assertTrue(np.isfinite(summary["RMSE_mm_day"]))

    def test_checkpoint_restores_predictions_and_metadata(self):
        torch.manual_seed(5)
        model = DeepESD(15, 5, (3, 4))
        before = downscale(model, self.x, self.prep)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, self.prep, {"period": ["1981-01-01", "1981-01-06"]})
            restored, prep, metadata = load_checkpoint(path)
            after = downscale(restored, self.x, prep)
        xr.testing.assert_identical(before, after)
        self.assertEqual(metadata["period"][1], "1981-01-06")
        self.assertTrue(bool((before >= 0).where(self.prep.land_mask, True).all()))

    def test_training_returns_best_validation_weights(self):
        torch.manual_seed(4)
        torch.set_num_threads(2)
        model = DeepESD(15, 5, (3, 4))
        x, y = self.prep.inputs(self.x), self.prep.targets(self.y)
        model.initialize_from_mean(y[:6].mean(0))
        history, info = train_model(model, x[:6], y[:6], x[6:], y[6:], epochs=5, lr=.02, patience=2)
        with torch.no_grad():
            loss = float(torch.nn.functional.mse_loss(model(torch.from_numpy(x[6:])), torch.from_numpy(y[6:])))
        self.assertAlmostEqual(loss, info["best_validation_mse"], places=5)
        self.assertLessEqual(info["best_validation_mse"], info["initial_validation_mse"])
        self.assertLessEqual(len(history), 5)


class RawPreprocessingTests(unittest.TestCase):
    def raw_files(self, directory, missing_hour=False):
        """Two monthly files spanning Jan 31 and Feb 1 with known accumulations."""
        xpaths, ypaths = [], []
        units = {"u": "m s**-1", "v": "m s**-1", "q": "kg kg**-1", "t": "K", "z": "m**2 s**-2"}
        for i, day in enumerate(["1981-01-31", "1981-02-01"]):
            xtime = pd.date_range(day, periods=4, freq="6h")
            if missing_hour and i == 0:
                xtime = xtime.delete(1)
            x = xr.Dataset({v: (("valid_time", "pressure_level", "latitude", "longitude"),
                                np.full((len(xtime), 3, 2, 2), 1.0 + i), {"units": units[v]}) for v in units},
                            coords={"valid_time": xtime, "pressure_level": [850, 700, 500],
                                    "latitude": [42., 41.], "longitude": [28., 29.]})
            x.pressure_level.attrs["units"] = "hPa"
            rain = np.full((24, 2, 2), .0005)
            rain[0] = .003 if i == 0 else .012
            y = xr.Dataset({"tp": (("valid_time", "latitude", "longitude"), rain,
                                    {"units": "m", "GRIB_stepType": "accum"}),
                            "t2m": (("valid_time", "latitude", "longitude"), np.full((24, 2, 2), 280.), {"units": "K"})},
                            coords={"valid_time": pd.date_range(day, periods=24, freq="h"),
                                    "latitude": [42., 41.], "longitude": [28., 29.]})
            xp, yp = Path(directory) / f"x{i}.nc", Path(directory) / f"y{i}.nc"
            x.to_netcdf(xp)
            y.to_netcdf(yp)
            xpaths.append(xp)
            ypaths.append(yp)
        return xpaths, ypaths

    def test_next_month_midnight_supplies_previous_day(self):
        with tempfile.TemporaryDirectory() as directory:
            x, y, audit = prepare_daily(*self.raw_files(directory))
        self.assertEqual(audit["days"], 1)
        self.assertEqual(audit["start"], "1981-01-31")
        self.assertEqual(audit["excluded_predictor_dates"], ["1981-02-01"])
        np.testing.assert_allclose(y.pr.values, 12.)
        np.testing.assert_allclose(y.tasmax.values, 6.85)
        self.assertEqual(list(x.data_vars), PRED_VARS)

    def test_partial_raw_day_is_not_silently_averaged(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.raw_files(directory, missing_hour=True)
            with self.assertRaisesRegex(ValueError, "örneklemesi"):
                prepare_daily(*paths)

    def test_cds_zip_is_read_as_netcdf(self):
        with tempfile.TemporaryDirectory() as directory:
            xp, _ = self.raw_files(directory)
            archive = Path(directory) / "download.nc"
            with zipfile.ZipFile(archive, "w") as z:
                z.write(xp[0], "nested/data_0.nc")
            xr.testing.assert_identical(read_netcdf(xp[0]), read_netcdf(archive))

    def test_missing_raw_value_is_not_hidden_by_daily_average(self):
        with tempfile.TemporaryDirectory() as directory:
            xp, yp = self.raw_files(directory)
            broken = read_netcdf(xp[0])
            broken.u.values[1, 0, 0, 0] = np.nan
            broken.to_netcdf(xp[0])
            with self.assertRaisesRegex(ValueError, "Ham ERA5 girdisinde"):
                prepare_daily(xp, yp)

    def test_missing_hourly_temperature_is_not_hidden_by_daily_max(self):
        with tempfile.TemporaryDirectory() as directory:
            xp, yp = self.raw_files(directory)
            broken = read_netcdf(yp[0])
            broken.t2m.values[1, 0, 0] = np.nan
            broken.to_netcdf(yp[0])
            with self.assertRaisesRegex(ValueError, "saatlik veri eksik"):
                prepare_daily(xp, yp)


if __name__ == "__main__":
    unittest.main()
