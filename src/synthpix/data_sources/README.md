# Data Sources

This directory contains the implementations of the `DataSource` abstractions used to feed data into the `grain`-based data loading pipeline.

## Overview

The core abstraction is the `FileDataSource` (inheriting from `grain.RandomAccessDataSource`). Unlike the legacy schedulers which were often stateful (maintaining open file handles or internal pointers), these data sources are designed to be **stateless** and support random access by index. This makes them compatible with `grain`'s multiprocessing and shuffling capabilities.

## Key Components

-   **`FileDataSource`**: The abstract base class that handles recursive file discovery. It maps an integer index `[0, len(dataset))` to a specific file on disk.
-   **`HDF5DataSource`**: Implementation for loading flow fields from `.h5` files.
-   **`MATDataSource`**: Implementation for loading flow fields from `.mat` files.
-   **`EpisodicDataSource`**: A wrapper that groups files into temporal episodes, allowing the system to sample consecutive sequences of frames (e.g., for temporal training).

## HDF5 Data Source Implementation

The `HDF5DataSource` has a notable difference from the legacy `HDF5FlowFieldScheduler`:

-   **Statelessness**: It does not keep the HDF5 file open between calls. Every call to `__getitem__` opens the file, reads the data, and closes the file.
-   **Performance**: This design ensures the data source is pickleable (required for `grain` multi-worker loading) and thread-safe.
-   **Full Field Loading**: A significant difference from the legacy implementation is that the `HDF5DataSource` loads the entire 3D field at once, whereas the previous scheduler loaded only a single slice at a time.

## Usage

These data sources are typically not instantiated directly by the end-user but are created via the `synthpix.make` entry point, which wraps them in a `grain.DataLoader`.
