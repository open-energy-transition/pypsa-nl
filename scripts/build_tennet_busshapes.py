# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-FileCopyrightText: Open Energy Transition gGmbH
#
# SPDX-License-Identifier: MIT

import geopandas as gpd
import pandas as pd

from shapely.geometry import Polygon, MultiPolygon
from shapely import union_all

from scripts._helpers import (
    set_scenario_config,
)

def largest_polygon(geom):
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return geom

def fill_gdf_holes(gdf: gpd.GeoDataFrame, gdf_nl: gpd.GeoDataFrame):

    merged = union_all(gdf.geometry)

    # Extract holes
    holes = [
        Polygon(hole)
        for poly in (merged.geoms if merged.geom_type == "MultiPolygon" else [merged])
        for hole in poly.interiors
    ]

    gdf_holes = gpd.GeoDataFrame(geometry=holes, crs=gdf.crs)
    gdf_holes["orig_area"] = gdf_holes.to_crs(3857).geometry.area

    # Prevent overlays with existing shapes (island within shapes)
    gdf_holes = gpd.overlay(
        gdf_holes,
        gdf,
        how="difference"
    )

    # Filter out large bodies of water
    gdf_holes = gpd.overlay(
        gdf_nl,
        gdf_holes,
        how="intersection"
    )

    # Retain only the largest polygon
    gdf_holes["geometry"] = gdf_holes.geometry.apply(largest_polygon)

    # Drop patches with area loss larger than 99%
    gdf_holes["new_area"] = gdf_holes.to_crs(3857).geometry.area
    gdf_holes["subtract_ratio"] = gdf_holes["new_area"]/ gdf_holes["orig_area"]
    gdf_holes = gdf_holes[gdf_holes["subtract_ratio"] > 0.01]

    gdf_holes = gdf_holes[["geometry"]].reset_index(drop=True)

    patch_codes = [f"PATCH{i}" for i in gdf_holes.index]

    gdf_holes["statcode"] = patch_codes
    gdf_holes["BU_code"] = patch_codes
    gdf_holes["id"] = None
    gdf_holes["archetype"] = 0
    gdf_holes = gdf_holes.set_index("statcode")

    gdf_updated = pd.concat([gdf,gdf_holes])

    return gdf_updated


def dissolve_shapes_using_lasso(gdf: gpd.GeoDataFrame, gdf_lasso: gpd.GeoDataFrame):

    # Use a projected CRS for accurate area calculations
    gdf_proj = gdf.to_crs(3857)
    gdf_lasso_proj = gdf_lasso.to_crs(3857)

    # Save original polygon index and area
    gdf_proj = gdf_proj.reset_index(names="orig_idx")
    gdf_proj["orig_area"] = gdf_proj.geometry.area

    # Compute intersections
    intersection = gpd.overlay(
        gdf_lasso_proj,
        gdf_proj,
        how="intersection"
    )

    # Calculate overlap ratio relative to the original polygon
    intersection["overlap_ratio"] = (
        intersection.geometry.area / intersection["orig_area"]
    )

    # Keep only intersections covering >= 50%
    matches = intersection[intersection["overlap_ratio"] >= 0.5]

    for idx, row in matches.iterrows():
        gdf_idx = row["BU_code"]
        gdf.loc[gdf_idx, "pockets"] = row["name"]


    # Count archetype frequencies per pocket
    archetype_counts = (
        gdf.groupby(["pockets", "archetype"])
        .size()
        .unstack(fill_value=0)
        .add_prefix("archetype_")
    )

    # Dissolve geometries by pockets
    pocket_gdf = gdf.dissolve(by="pockets")[["geometry"]]

    # Add archetype frequency columns
    pocket_gdf = pocket_gdf.join(archetype_counts)

    return pocket_gdf


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

    # 1. Retrieve relevant GeoDataFrames
    gdf_buurten = gpd.read_file(snakemake.input.archetypen_buurten).set_index("statcode")
    gdf = gpd.read_file(snakemake.input.admin_shapes)
    gdf_traces = gpd.read_file(snakemake.input.pockets_traces)

    # 2. Split NL from the admin shapes
    gdf = gdf.rename(columns={"admin": "name"})[["name", "country", "geometry"]]
    gdf_nl = gdf[gdf["country"] == "NL"]
    gdf_wo_nl = gdf[gdf["country"] != "NL"]

    # 3. Fill the gaps in the Buurten gdf but stick only to NL land shapes 
    gdf_buurten_patch = fill_gdf_holes(gdf_buurten, gdf_nl)

    # 4. Dissolve the Buurten using the pocket traces, save them for other uses
    gdf_pockets = dissolve_shapes_using_lasso(gdf_buurten_patch, gdf_traces)
    gdf_pockets.to_file("data/ISIE/pockets_with_archetypes.geojson")

    # 5. Merged NL that includes the pockets with other countries and save them as busshapes
    gdf_nl_pockets = gdf_pockets[["geometry"]].reset_index().rename(columns={"pockets": "name"})
    gdf_nl_pockets["country"] = "NL"

    gdf_new = pd.concat([gdf_wo_nl, gdf_nl_pockets]).reset_index(drop=True)
    gdf_new.to_file(snakemake.output.busshape)
