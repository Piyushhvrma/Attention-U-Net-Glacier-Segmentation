# Dataset Guide: Sentinel-2 & RGI 14

## Overview
This project uses **8-channel multispectral patches** derived from Copernicus Sentinel-2 Level-2A surface reflectance imagery and Randolph Glacier Inventory (RGI 7.0/6.0 Region 14: South Asia West) vector outlines.

### 8-Channel Specification
1. `B2` (Blue) - ~490 nm
2. `B3` (Green) - ~560 nm
3. `B4` (Red) - ~665 nm
4. `B8` (NIR) - ~842 nm
5. `B11` (SWIR1) - ~1610 nm
6. `B12` (SWIR2) - ~2190 nm
7. `NDSI` - Normalized Difference Snow Index (`(B3 - B11) / (B3 + B11)`)
8. `NDVI` - Normalized Difference Vegetation Index (`(B8 - B4) / (B8 + B4)`)
