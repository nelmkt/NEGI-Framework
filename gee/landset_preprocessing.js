// ============================================================
// Jeddah LST / NDVI / NDBI dataset builder — v5
//
// Builds on v4 (mixed-pixel vegetation purity flagging).
// Everything from v4 is unchanged below except where marked
// "NEW (v5)".
//
// v4 computed and exported clean_vegetation_flag as a column but
// never actually used it to filter anything — the "purified" vs
// "raw" comparison depended on a downstream Python toggle that may
// or may not have been exercised. That's silent and easy to miss.
// v5 fixes this two ways:
//
// NEW (v5) #1 — Purity filter is now actually applied at export
// time, not just computed. The script exports TWO tables from ONE
// GEE run: 'raw' (all valid pixels, current behavior) and
// 'purified' (mixed vegetation pixels dropped). Both carry
// clean_vegetation_flag anyway, so nothing is lost, but you no
// longer have to trust that a downstream Python flag was set
// correctly — the purified table is purified by construction.
// Sample size for the purified export is oversampled up front
// since filtering removes points; the print statements report the
// actual purified sample size so you can confirm it's not
// underpowered before running the model.
//
// NEW (v5) #2 — Emissivity diagnostic. ST_B10 in Collection 2
// Level 2 is not raw brightness temperature: USGS's single-channel
// algorithm derives per-pixel emissivity partly from ASTER GED
// blended with an NDVI-based vegetation-fraction term before this
// band is delivered. That means NDVI and LST are not fully
// independent measurements at the source-product level, and a
// positive NDVI-LST slope in the downstream model could partly
// reflect that upstream coupling rather than a purely physical
// effect. This script cannot remove that coupling (doing so
// requires reimplementing the ST algorithm with an
// NDVI-independent emissivity source, out of scope here), but it
// now exports the delivered ST_EMIS (and ST_EMSD) bands alongside
// NDVI so the downstream analysis can directly check how much of
// the raw NDVI-LST correlation lines up with emissivity variation.
// If the NDVI-emissivity correlation is high, that's evidence the
// fitted slope is at least partly a source-product artifact, not
// solely a physical greening effect — this should be reported as a
// limitation regardless of what the check shows.
// ============================================================

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

  // NEW (v5): scale the emissivity QA bands the same way the
  // thermal band is scaled, using their documented C2 L2 scale
  // factors, so they're in physical units (0-1) on export.
  var emis = image.select('ST_EMIS').multiply(0.0001);
  var emisStdDev = image.select('ST_EMSD').multiply(0.0001);

  image = image
      .addBands(optical, null, true)
      .addBands(thermal, null, true)
      .addBands(emis, null, true)
      .addBands(emisStdDev, null, true)
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
  // the fitted NDVI coefficient downstream. See v5 header note: the
  // ST_EMIS/ST_EMSD bands are now exported specifically so this can
  // be checked quantitatively rather than just asserted.
  var lst = image.select('ST_B10')
      .subtract(273.15)
      .rename('LST');

  return image.addBands([ndvi, ndbi, ndwi, lst]);
}

// Composite: per-image indices computed first, then medianed —
// i.e. median(NDVI_t), not NDVI computed from already-medianed bands.
var composite = collection
    .map(preprocess)
    .select(['NDVI', 'NDBI', 'NDWI', 'LST', 'ST_EMIS', 'ST_EMSD'])
    .median();

var ndvi = composite.select('NDVI');
var ndbi = composite.select('NDBI');
var ndwi = composite.select('NDWI');
var lst  = composite.select('LST');
var emisBand = composite.select('ST_EMIS');
var emisStdDevBand = composite.select('ST_EMSD');

var elevation = ee.Image("USGS/SRTMGL1_003").rename('Elevation');

// --- Land/water mask -----------------------------------------
// Water: NDWI > 0. Land: everything else, eroded 300m from any
// water edge to avoid mixed coastal/wadi pixels.
var waterMask = ndwi.gt(0);
var landMask = waterMask.not();
landMask = landMask.focal_min({radius: 300, units: 'meters'});

// --- Quality mask ----------------------------------------------
// LST ceiling at 60°C so the observed hot tail (~55°C max) is not
// truncated. Lower bound and index bounds unchanged.
var qualityMask =
    lst.gt(20)
    .and(lst.lt(60))
    .and(ndvi.gte(-0.2))
    .and(ndvi.lte(0.9))
    .and(ndbi.gte(-0.5))
    .and(ndbi.lte(0.5));

var finalMask = landMask.and(qualityMask);

// ============================================================
// v4: mixed-pixel vegetation purity flagging (unchanged)
// ============================================================
var VEG_NDVI_THRESHOLD    = 0.2;   // matches the existing "tail pixel" cut used below
var MAX_LOCAL_NDVI_STDDEV = 0.15;  // heterogeneity ceiling for "homogeneous" vegetation
var MIN_PATCH_PIXELS      = 9;     // ~3x3 30m pixels (~8,100 m²); below this = isolated/edge feature
var LOCAL_STDDEV_RADIUS_M = 45;    // neighborhood radius for the homogeneity check
var PATCH_COUNT_MAX_SIZE  = 200;   // cap for connectedPixelCount (perf guard; large parks saturate here, which is fine)

var ndviLocalStddev = ndvi
    .reduceNeighborhood({
      reducer: ee.Reducer.stdDev(),
      kernel: ee.Kernel.square({radius: LOCAL_STDDEV_RADIUS_M, units: 'meters'}),
    })
    .rename('NDVI_local_stddev');

var vegMask = ndvi.gt(VEG_NDVI_THRESHOLD);
var vegPatchPixelCount = vegMask.selfMask()
    .connectedPixelCount({maxSize: PATCH_COUNT_MAX_SIZE, eightConnected: true})
    .rename('veg_patch_pixel_count');

var vegPatchPixelCountFilled = vegPatchPixelCount.unmask(0);
var passesHomogeneity = ndviLocalStddev.lte(MAX_LOCAL_NDVI_STDDEV);
var passesPatchSize   = vegPatchPixelCountFilled.gte(MIN_PATCH_PIXELS);

var cleanVegetationFlag = vegMask
    .not()  // start true (1) for non-vegetated pixels
    .or(passesHomogeneity.and(passesPatchSize))  // OR: vegetated AND passes both checks
    .rename('clean_vegetation_flag');

// ============================================================
// End v4 addition
// ============================================================

var gridSizeDeg = 0.01;
var lonLat = ee.Image.pixelLonLat();
var blockId = lonLat.select('longitude').divide(gridSizeDeg).floor()
    .multiply(100000)
    .add(lonLat.select('latitude').divide(gridSizeDeg).floor())
    .rename('block_id');

var stack = ee.Image.cat([
  ndvi, ndbi, lst, elevation, blockId,
  ndviLocalStddev, vegPatchPixelCountFilled, cleanVegetationFlag,
  emisBand, emisStdDevBand,   // NEW (v5)
]).updateMask(finalMask);

Map.centerObject(jeddah, 10);

Map.addLayer(lst, {
  min: 20, max: 55,
  palette: ['blue', 'cyan', 'green', 'yellow', 'red']
}, 'LST');

Map.addLayer(landMask.selfMask(), {palette: ['yellow']}, 'Land Mask');
Map.addLayer(waterMask.selfMask(), {palette: ['blue']}, 'Water');

Map.addLayer(
  cleanVegetationFlag.updateMask(vegMask).selfMask(),
  {min: 0, max: 1, palette: ['red', 'green']},
  'Vegetation purity (green=clean, red=impure/mixed)'
);

// NEW (v5): visualize emissivity so it can be eyeballed against the
// NDVI layer for obvious spatial coupling before doing the formal
// downstream correlation check.
Map.addLayer(
  emisBand.updateMask(finalMask),
  {min: 0.95, max: 1.0, palette: ['orange', 'blue']},
  'Emissivity (ST_EMIS)'
);

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

// NEW (v5): quick sanity check on the NDVI-emissivity coupling
// concern, computed directly in GEE so it's visible before export
// rather than only discoverable later in Python.
print('NDVI vs Emissivity correlation (masked, land-only):',
  ndvi.updateMask(finalMask)
    .addBands(emisBand.updateMask(finalMask))
    .reduceRegion({
      reducer: ee.Reducer.pearsonsCorrelation(),
      geometry: jeddahLand,
      scale: 30,
      maxPixels: 1e10
    })
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

var highNdviClean = highNdvi.and(cleanVegetationFlag.eq(1));
print('Of those, pixels passing the purity filter (land-only):',
  highNdviClean.updateMask(finalMask).reduceRegion({
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

// ============================================================
// NEW (v5): export RAW and PURIFIED tables from one run.
// ============================================================
// Raw sample size unchanged from v4 (30000). Purified is
// oversampled up front — the purity filter drops roughly the same
// fraction of pixels regardless of where you sample from, so
// pulling more raw candidate points before filtering keeps the
// final purified table close to full size rather than silently
// underpowered.
var RAW_NUM_PIXELS = 30000;
var PURIFIED_CANDIDATE_NUM_PIXELS = 45000; // oversample; final count reported below
var PURIFIED_TARGET_NUM_PIXELS = 30000;    // trimmed to this after filtering, if enough survive

var rawSamples = stack.sample({
  region: jeddahLand,
  scale: 30,
  numPixels: RAW_NUM_PIXELS,
  seed: 42,
  geometries: true,
  tileScale: 4
});

rawSamples = rawSamples.map(function(f) {
  var coords = f.geometry().coordinates();
  return f.set({'lon': coords.get(0), 'lat': coords.get(1)});
});

print('Raw sample size:', rawSamples.size());
print(rawSamples.limit(5));

Export.table.toDrive({
  collection: rawSamples,
  description: 'Jeddah_LST_Dataset_2023_raw',
  fileFormat: 'CSV'
});

// Purified table: same stack, same seed (for comparability), but
// sampled with the purity mask applied so only clean_vegetation_flag
// == 1 pixels can be drawn. Non-vegetated pixels are unaffected
// (they're flagged clean by definition, per the v4 header note).
var purifiedStack = stack.updateMask(cleanVegetationFlag.eq(1));

var purifiedCandidates = purifiedStack.sample({
  region: jeddahLand,
  scale: 30,
  numPixels: PURIFIED_CANDIDATE_NUM_PIXELS,
  seed: 42,
  geometries: true,
  tileScale: 4
});

var purifiedSampleSize = purifiedCandidates.size();
print('Purified candidate sample size (before trim):', purifiedSampleSize);

// Trim to the target size for a fair train/holdout split comparison
// against the raw table, if enough points survived filtering;
// otherwise export everything that survived and flag it.
var purifiedSamples = ee.FeatureCollection(
  ee.Algorithms.If(
    purifiedSampleSize.gte(PURIFIED_TARGET_NUM_PIXELS),
    purifiedCandidates.limit(PURIFIED_TARGET_NUM_PIXELS),
    purifiedCandidates
  )
);

purifiedSamples = purifiedSamples.map(function(f) {
  var coords = f.geometry().coordinates();
  return f.set({'lon': coords.get(0), 'lat': coords.get(1)});
});

print('Purified sample size (final, after trim):', purifiedSamples.size());
print('WARNING: if this is much smaller than', PURIFIED_TARGET_NUM_PIXELS,
  '- raise PURIFIED_CANDIDATE_NUM_PIXELS and rerun before trusting a purified-vs-raw comparison.');
print(purifiedSamples.limit(5));

Export.table.toDrive({
  collection: purifiedSamples,
  description: 'Jeddah_LST_Dataset_2023',
  fileFormat: 'CSV'
});
// ============================================================
// End v5 export changes
// ============================================================

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
