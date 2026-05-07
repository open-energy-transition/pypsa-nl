# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-FileCopyrightText: Open Energy Transition gGmbH
#
# SPDX-License-Identifier: MIT

import geopandas as gpd
import pandas as pd

from scripts._helpers import (
    set_scenario_config,
)


def merge_small_polygons(gdf, area_threshold, prefer_touching=True, fix_invalid=True):
    """
    Merge polygons smaller than area_threshold into their closest neighbor.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input polygons (must have a valid geometry column)
    area_threshold : float
        Minimum area; polygons smaller than this will be merged
    prefer_touching : bool
        If True, merge with touching neighbors first; otherwise use nearest only
    fix_invalid : bool
        If True, apply buffer(0) to fix invalid geometries after merges

    Returns
    -------
    GeoDataFrame
        Cleaned GeoDataFrame with small polygons merged
    """

    # Work on a copy
    gdf = gdf.copy()
    gdf = gdf.to_crs(3857)

    # Ensure projected CRS (area in meters, not degrees)
    # if gdf.crs and gdf.crs.is_geographic:
    #     raise ValueError("GeoDataFrame must be in a projected CRS (not EPSG:4326)")

    def find_best_neighbor(idx):
        geom = gdf.loc[idx].geometry

        # Try touching neighbors first
        if prefer_touching:
            touching = gdf[gdf.geometry.touches(geom)].drop(idx, errors="ignore")
            if not touching.empty:
                # choose the largest neighbor (more stable)
                return touching["Shape__Area"].idxmax()

        # Fallback: nearest neighbor
        distances = gdf.geometry.distance(geom)
        distances = distances.drop(idx)
        return distances.idxmin()

    # Process smallest polygons first
    while True:
        small = gdf[gdf["Shape__Area"] < area_threshold]

        if small.empty:
            break

        # pick smallest polygon
        idx = small["Shape__Area"].idxmin()

        # find neighbor
        neighbor_idx = find_best_neighbor(idx)

        # merge geometries
        new_geom = gdf.loc[idx].geometry.union(gdf.loc[neighbor_idx].geometry)

        # assign merged geometry to neighbor
        gdf.at[neighbor_idx, "geometry"] = new_geom

        # drop the small polygon
        gdf = gdf.drop(idx)

        # recompute area
        gdf["Shape__Area"] = gdf.geometry.area

        # optionally fix invalid geometries
        if fix_invalid:
            gdf["geometry"] = gdf.buffer(0)

    return gdf.to_crs(4326)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_tennet_busshapes",
            configfiles="config/config.nl-core.yaml",
            clusters=20,
            base_network="osm",
        )
    set_scenario_config(snakemake)

    # 1. Retrieve and build the Tennet subdivision
    gdf_nl = gpd.read_file(snakemake.input.pocketsWGS)
    gdf_nl["country"] = "NL"

    # Remove missing province
    gdf_nl = gdf_nl[~gdf_nl.provincie.isna()]

    # Merge small regions
    gdf_nl = merge_small_polygons(gdf_nl, area_threshold=1e8)

    # Invert the dictionary: objectid -> network name
    tennet_network = snakemake.params.tennet_network

    id_to_network = {
        obj_id: network_name
        for network_name, ids in tennet_network.items()
        for obj_id in ids
    }

    gdf_nl["name"] = gdf_nl.OBJECTID.map(id_to_network).fillna(gdf_nl["OBJECTID"])
    gdf_nl = gdf_nl.dissolve("name")[["country", "geometry"]].reset_index()

    # 2. Retrieve the admin_shapes and take out NL
    gdf = gpd.read_file(snakemake.input.admin_shapes).rename(columns={"admin": "name"})[
        ["name", "country", "geometry"]
    ]
    gdf = gdf[gdf["country"] != "NL"]

    # 3. Combine and save
    gdf_new = pd.concat([gdf, gdf_nl]).reset_index(drop=True)
    gdf_new.to_file(snakemake.output.busshape)
