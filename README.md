# CODE_PAPER1
Code and Tools availability statement
Open-source tools were only used for the manuscript. The data extraction was carried out using Google Earth Engine (https://code.earthengine.google.com/). The SVR modelling was carried out using Anaconda which can be downloaded from-  https://www.anaconda.com/download, using the Jupyter module. The Maps were prepared in QGIS which is an open-source software and can be downloaded for free from- https://qgis.org/download/. 
I. GOOGLE EARTH ENGINE SCRIPTS
1. CHIRPS Precipitation Data Extraction
var chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
  .filterDate('2021-05-01', '2023-05-31')
  .select('precipitation')
  .mean()
  .clip(regionOfInterest);

Export.image.toDrive({
  image: chirps,
  description: 'CHIRPS_Mean_2021_2023',
  scale: 5000,
  region: regionOfInterest
});
2. MODIS ET (MOD16A2) Data Extraction
var modisET = ee.ImageCollection("MODIS/006/MOD16A2")
  .filterDate('2021-05-01', '2023-05-31')
  .select('ET')
  .mean()
  .clip(regionOfInterest);
Export.image.toDrive({
  image: modisET,
  description: 'MODIS_ET_Mean_2021_2023',
  scale: 500,
  region: regionOfInterest
});
3. SMAP Soil Moisture Data Extraction
var smap = ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture")
  .filterDate('2021-05-01', '2023-05-31')
  .select('ssm')
  .mean()
  .clip(regionOfInterest);

Export.image.toDrive({
  image: smap,
  description: 'SMAP_Mean_2021_2023',
  scale: 10000,
  region: regionOfInterest
});
II. PYTHON CODE (FOR SVR MODELING & EVALUATION)
1. Data Preprocessing and Resampling
import rasterio
from rasterio.enums import Resampling
import numpy as np
def resample_raster(src_path, dst_path, scale_factor):
    with rasterio.open(src_path) as src:
        data = src.read(
            out_shape=(
                src.count,
                int(src.height * scale_factor),
                int(src.width * scale_factor)
            ),
            resampling=Resampling.nearest
        )
        profile = src.profile
        profile.update({
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": src.transform * src.transform.scale(
                (src.width / data.shape[-1]),
                (src.height / data.shape[-2])
            )
        })
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)

2. Support Vector Regression (SVR) with Scikit-learn
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
# X: CHIRPS - MODIS ET difference, y: SMAP soil moisture
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
svr_model = SVR(kernel='linear', C=1.8, epsilon=0.005)
svr_model.fit(X_train, y_train)
y_pred = svr_model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
#Notes:
#•	The manuscript used linear, polynomial, RBF, and sigmoid kernels — these can be toggled using the kernel argument in the SVR instantiation.
#•	Downscaling and reprojection to a 500 m resolution was done using nearest neighbor resampling.
#•	The workflow involved GEE for data preparation and Python (scikit-learn, rasterio) for modeling and evaluation.
