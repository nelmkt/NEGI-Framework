// Attach independent land-cover labels to every pixel of the NEGI dataset, so the
// pipeline can test what NDBI measures in Jeddah (built-up vs bare/dry vs moist).
//
// 1. Upload Jeddah_points_for_gee.csv (just lon, lat for all 28,990 pixels) to Earth Engine as a table asset
//    (Assets > New > CSV file). Under "Advanced options" set X column = lon and Y column = lat
//    (Earth Engine looks for "longitude"/"latitude" by default). Keep lon and lat as properties:
//    the pipeline joins on them, and it stops with an error if fewer than half the pixels match.
// 2. Set ASSET below to that asset's path and run. It exports a CSV with
//    lon, lat (join keys) + four new columns to your Drive folder "negi".
// 3. Run the pipeline with   --landcover path\to\Jeddah_landcover.csv
//
// Columns added
//   worldcover        ESA WorldCover v200 (2021, 10 m) class code at the point
//                     (10 tree, 20 shrub, 30 grass, 40 crop, 50 built-up, 60 bare/sparse,
//                      80 water, 90 wetland, 95 mangrove)
//   wc_built_share    share of 10 m WorldCover cells classed built-up (50) within 30 m (3x3 Landsat-scale window)
//   wc_bare_share     same for bare/sparse (60)
//   ghsl_built_frac   GHSL built-up surface 2020 (100 m): built m² / cell area (0–1)

var ASSET = 'users/YOUR_USER/Jeddah_LST_Dataset_2023_raw';   // <-- change
var pts = ee.FeatureCollection(ASSET);

var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var built = wc.eq(50).rename('wc_built_share');
var bare = wc.eq(60).rename('wc_bare_share');
var win = ee.Reducer.mean();
var shares = built.addBands(bare)
  .reduceNeighborhood({reducer: win, kernel: ee.Kernel.square(15, 'meters')})
  .rename(['wc_built_share', 'wc_bare_share']);

var ghsl = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020').select('built_surface')
  .divide(10000).clamp(0, 1).rename('ghsl_built_frac');   // m² per 100 m cell -> fraction

var stack = wc.rename('worldcover').addBands(shares).addBands(ghsl);

var out = stack.reduceRegions({collection: pts, reducer: ee.Reducer.first(), scale: 10})
  .map(function (f) {
    // lon/lat from the uploaded columns if kept as properties, otherwise from the point geometry
    var c = f.geometry().coordinates();
    var has = f.propertyNames();
    return ee.Feature(null, {
      'lon': ee.Algorithms.If(has.contains('lon'), f.get('lon'), c.get(0)),
      'lat': ee.Algorithms.If(has.contains('lat'), f.get('lat'), c.get(1)),
      'worldcover': f.get('worldcover'),
      'wc_built_share': f.get('wc_built_share'),
      'wc_bare_share': f.get('wc_bare_share'),
      'ghsl_built_frac': f.get('ghsl_built_frac')
    });
  });

Export.table.toDrive({
  collection: out, description: 'Jeddah_landcover', folder: 'negi', fileFormat: 'CSV',
  selectors: ['lon', 'lat', 'worldcover', 'wc_built_share', 'wc_bare_share', 'ghsl_built_frac']
});
