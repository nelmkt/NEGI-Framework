var jeddah = ee.Geometry.Rectangle([39.0, 21.2, 39.4, 21.8]);
function maskL8(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0)
      .and(qa.bitwiseAnd(1 << 4).eq(0))
      .and(qa.bitwiseAnd(1 << 5).eq(0));
  return image.updateMask(mask);
}
function applyScaleFactors(image) {
  var opticalBands = image.select('SR_B.*')
      .multiply(0.0000275)
      .add(-0.2);
  var thermalBand = image.select('ST_B10')
      .multiply(0.00341802)
      .add(149.0);
  return image
      .addBands(opticalBands, null, true)
      .addBands(thermalBand, null, true);
}
var image = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    .filterBounds(jeddah)
    .filterDate("2023-01-01", "2023-12-31")
    .map(maskL8)
    .map(applyScaleFactors)
    .median();
var ndvi = image.normalizedDifference(["SR_B5", "SR_B4"])
    .rename("NDVI");
var ndbi = image.normalizedDifference(["SR_B6", "SR_B5"])
    .rename("NDBI");
var ndwi = image.normalizedDifference(["SR_B3", "SR_B5"])
    .rename("NDWI");
var lst = image.select("ST_B10")
    .subtract(273.15)
    .rename("LST");
var landMask = ndwi.lt(0);
var elevation = ee.Image("USGS/SRTMGL1_003")
    .rename("Elevation");
var stack = ndvi
    .addBands(ndbi)
    .addBands(lst)
    .addBands(elevation)
    .updateMask(landMask);
Map.centerObject(jeddah,10);
Map.addLayer(ndvi,{min:-0.2,max:0.6},"NDVI");
Map.addLayer(ndbi,{min:-0.5,max:0.5},"NDBI");
Map.addLayer(lst,{min:20,max:55},"LST");
Map.addLayer(elevation,{min:0,max:200},"Elevation");
Map.addLayer(landMask.selfMask(),{palette:["green"]},"Land Mask");
var samples = stack.sample({
    region:jeddah,
    scale:30,
    numPixels:15000,
    seed:42,
    geometries:true
});
Export.table.toDrive({
    collection:samples,
    description:"Jeddah_NDVI_NDBI_LST_dataset_Masked",
    fileFormat:"CSV"
});
