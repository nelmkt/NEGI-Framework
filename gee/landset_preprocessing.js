 
var jeddah = ee.Geometry.Rectangle([39.0, 21.2, 39.4, 21.8]);
 
var saudiBoundary = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
  .filter(ee.Filter.eq('country_na', 'Saudi Arabia'))
  .geometry();
 
var jeddahLand = jeddah.intersection(saudiBoundary, ee.ErrorMargin(1));
 
var collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
  .filterBounds(jeddah)
  .filterDate("2023-01-01", "2023-12-31")
  .filter(ee.Filter.lt('CLOUD_COVER', 20));
 
print('Images used:', collection.size());
 
function preprocess(image) {
 
  var qa = image.select('QA_PIXEL');
  var sat = image.select('QA_RADSAT');
 
  var mask = qa.bitwiseAnd(1 << 1).eq(0)   // dilated cloud
      .and(qa.bitwiseAnd(1 << 2).eq(0))    // cirrus
      .and(qa.bitwiseAnd(1 << 3).eq(0))    // cloud
      .and(qa.bitwiseAnd(1 << 4).eq(0))    // cloud shadow
      .and(sat.eq(0));                     // no radiometric saturation
 
  var optical = image.select('SR_B.*')
      .multiply(0.0000275)
      .add(-0.2);
 
  var thermal = image.select('ST_B10')
      .multiply(0.00341802)
      .add(149.0);
 
  image = image
      .addBands(optical, null, true)
      .addBands(thermal, null, true)
      .updateMask(mask);
 
  var ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
  var ndbi = image.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI');
  var ndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
 
  // NOTE: ST_B10 here is USGS Collection 2 Level-2 Surface Temperature.
  // Its retrieval algorithm estimates emissivity partly from NDVI
  // (ASTER GED blended with an NDVI-based vegetation-fraction term)
  // before this band is delivered. That means NDVI is not fully
  // independent of LST at the source-product level — not a bug in
  // this script, but worth stating as a limitation when interpreting
  // the fitted NDVI coefficient downstream.
  var lst = image.select('ST_B10')
      .subtract(273.15)
      .rename('LST');
 
  return image.addBands([ndvi, ndbi, ndwi, lst]);
}
 
// Composite: per-image indices computed first, then medianed —
// i.e. median(NDVI_t), not NDVI computed from already-medianed bands.
var composite = collection
    .map(preprocess)
    .select(['NDVI', 'NDBI', 'NDWI', 'LST'])
    .median();
 
var ndvi = composite.select('NDVI');
var ndbi = composite.select('NDBI');
var ndwi = composite.select('NDWI');
var lst  = composite.select('LST');
 
var elevation = ee.Image("USGS/SRTMGL1_003").rename('Elevation');
 
// --- Land/water mask -----------------------------------------
// Water: NDWI > 0. Land: everything else, eroded 300m from any
// water edge to avoid mixed coastal/wadi pixels. (Removed the
// previous extra `.and(ndwi.lt(-0.1))` clause — see header note.)
var waterMask = ndwi.gt(0);
var landMask = waterMask.not();
landMask = landMask.focal_min({radius: 300, units: 'meters'});
 
// --- Quality mask ----------------------------------------------
// LST ceiling raised from 55°C to 60°C so the observed hot tail
// (~55°C max) is not truncated. Lower bound and index bounds
// unchanged. Re-check against the printed histogram/min-max below
// and tighten only if there's evidence of retrieval artifacts.
var qualityMask =
    lst.gt(20)
    .and(lst.lt(60))
    .and(ndvi.gte(-0.2))
    .and(ndvi.lte(0.9))
    .and(ndbi.gte(-0.5))
    .and(ndbi.lte(0.5));
 
var finalMask = landMask.and(qualityMask);
 
var gridSizeDeg = 0.01;
var lonLat = ee.Image.pixelLonLat();
var blockId = lonLat.select('longitude').divide(gridSizeDeg).floor()
    .multiply(100000)
    .add(lonLat.select('latitude').divide(gridSizeDeg).floor())
    .rename('block_id');
 
var stack = ee.Image.cat([ndvi, ndbi, lst, elevation, blockId]).updateMask(finalMask);
 
Map.centerObject(jeddah, 10);
 
Map.addLayer(lst, {
  min: 20, max: 55,
  palette: ['blue', 'cyan', 'green', 'yellow', 'red']
}, 'LST');
 
Map.addLayer(landMask.selfMask(), {palette: ['yellow']}, 'Land Mask');
Map.addLayer(waterMask.selfMask(), {palette: ['blue']}, 'Water');
 
print(
  ui.Chart.image.histogram({
    image: lst.updateMask(finalMask),
    region: jeddahLand,
    scale: 30,
    maxPixels: 1e8
  }).setOptions({title: 'LST Histogram (masked, land-only)'})
);
 
print('LST Statistics (masked, land-only)',
  lst.updateMask(finalMask).reduceRegion({
    reducer: ee.Reducer.minMax()
      .combine({reducer2: ee.Reducer.mean(), sharedInputs: true})
      .combine({reducer2: ee.Reducer.median(), sharedInputs: true}),
    geometry: jeddahLand,
    scale: 30,
    maxPixels: 1e10
  })
);
 
print(
  ui.Chart.image.histogram({
    image: ndvi.updateMask(finalMask),
    region: jeddahLand,
    scale: 30,
    maxPixels: 1e8
  }).setOptions({title: 'NDVI Histogram (masked, land-only)'})
);
 
var highNdvi = ndvi.updateMask(finalMask).gt(0.2);
Map.addLayer(highNdvi.selfMask(), {palette: ['magenta']}, 'NDVI > 0.2 (tail pixels)');
 
print('Pixel count with NDVI > 0.2 (land-only):',
  highNdvi.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: jeddahLand,
    scale: 30,
    maxPixels: 1e10
  })
);
 
var validPixels = stack.select('LST').mask().reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: jeddahLand,
  scale: 30,
  maxPixels: 1e10
});
print('Valid pixels:', validPixels);
 
var samples = stack.sample({
  region: jeddahLand,
  scale: 30,
  numPixels: 30000,
  seed: 42,
  geometries: true,
  tileScale: 4
});
 
samples = samples.map(function(f) {
  var coords = f.geometry().coordinates();
  return f.set({
    'lon': coords.get(0),
    'lat': coords.get(1)
  });
});
 
print('Sample size:', samples.size());
print(samples.limit(5));
 
Export.table.toDrive({
  collection: samples,
  description: 'Jeddah_LST_Dataset_2023_revised',
  fileFormat: 'CSV'
});
 
var lstMasked = lst.updateMask(finalMask);
var near30 = lstMasked.gt(29.5).and(lstMasked.lt(30.5));
 
Map.addLayer(near30.selfMask(), {palette: ['red']}, '~30C Pixels (land-only)');
 
var count30 = near30.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: jeddahLand,
  scale: 30,
  maxPixels: 1e10
});
print('Pixels near 30C (land-only):', count30);
