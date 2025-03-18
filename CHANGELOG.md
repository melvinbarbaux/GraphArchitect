# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

-----------------------------------------------------------------------------------------------------------------
## [0.1.1] - 2025-03-18

### Added
- Implemented all graph construction methods in `graph_construction` module.
  - Added K-Nearest Neighbors (KNN) graph construction.
  - Implemented Epsilon graph construction.
  - Developed Minimum Spanning Tree (MST) graph construction.
  - Created Anchor graph construction using KMeans clustering.
  - Added Delaunay triangulation-based graph construction.
- Introduced a unified `build_graph` function to dynamically construct graphs based on the chosen method.

### Technical
- Added type hints to all functions and methods.

-----------------------------------------------------------------------------------------------------------------
## [0.1.0] - 2025-03-17

### Added
- **Dataset Download System**: First implementation of a comprehensive dataset management system
  - Implemented download methods for sklearn datasets, torchvision datasets, and URL-based datasets
  - Added configuration via YAML for flexible dataset specifications
  - Ensured existing files are not redownloaded (caching mechanism)
  - Added directory structure logic to organize datasets by data type (tabular, image, audio, text, time_series)
  - Provided optional archive extraction for compressed datasets
  - Created a modular structure to easily extend with new data sources

### Technical
- Initial project setup with proper requirements
- Added .gitignore to prevent dataset files and other unnecessary files from being tracked

-----------------------------------------------------------------------------------------------------------------