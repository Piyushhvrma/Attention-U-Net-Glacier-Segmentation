import os
import argparse
import rasterio

def inspect_raster(tif_path):
    if not os.path.exists(tif_path):
        print(f"Error: File '{tif_path}' not found.")
        return
        
    with rasterio.open(tif_path) as src:
        print("="*60)
        print(f"GEOSPATIAL RASTER INSPECTION: {os.path.basename(tif_path)}")
        print("="*60)
        print(f"Width x Height:     {src.width} x {src.height}")
        print(f"Number of Bands:    {src.count}")
        print(f"Data Types:         {src.dtypes}")
        print(f"CRS (Projection):   {src.crs}")
        print(f"Spatial Bounds:     {src.bounds}")
        print(f"Pixel Resolution:   {src.res}")
        print(f"No-Data Value:      {src.nodatavals}")
        print("-" * 60)
        for i in range(1, src.count + 1):
            band_arr = src.read(i)
            print(f"Band {i} Stats -> Min: {band_arr.min():.4f}, Max: {band_arr.max():.4f}, Mean: {band_arr.mean():.4f}")
        print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Sentinel-2 GeoTIFF properties")
    parser.add_argument("--input", type=str, default="sample_scene.tif", help="Path to GeoTIFF file")
    args = parser.parse_args()
    inspect_raster(args.input)
