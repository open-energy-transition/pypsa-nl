# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-FileCopyrightText: Open Energy Transition gGmbH
#
# SPDX-License-Identifier: MIT
import logging

import numpy as np
import pandas as pd
import pypsa

from scripts._helpers import (
    configure_logging,
    set_scenario_config,
)

WEIGHTING_COLS = [
    "p_init",
    "p_max_pu",
    "p_min_pu",
    "p_nom",
    "p_nom_max",
    "p_nom_min",
    "p_nom_mod",
    "p_nom_opt",
    "p_nom_set",
    "p_set",
    "e_initial",
    "e_max_pu",
    "e_min_pu",
    "e_nom",
    "e_nom_max",
    "e_nom_min",
    "e_nom_mod",
    "e_nom_opt",
    "e_nom_set",
    "e_set",
]


def keep_country(
    n: pypsa.Network,
    countries: list[str],
    include_neighbours: bool = False,
) -> pypsa.Network:
    """
    Keep only the specified countries in the network.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to trim.
    countries : list[str]
        List of countries to keep.
    include_neighbours : bool, default False
        Whether to keep neighbouring buses that are still connected to the
        retained network.

    Returns
    -------
    pypsa.Network
        A copy of the network containing only the selected countries.
    """
    m = n.copy()

    keep_buses = m.buses[m.buses.country.isin(countries)].index

    # Remove all components not connected to those buses
    for c in m.components:
        bus_cols = c.static.columns[c.static.columns.str.startswith("bus")]
        if len(bus_cols) == 0:
            continue

        mask = c.static[bus_cols].isin(keep_buses).any(axis=1)
        if mask.any():
            m.remove(c.name, c.static.index[~mask])

    # Finally remove the buses not connected to anything
    if include_neighbours:
        drop_buses = set(m.buses.index)

        for c in m.components:
            if c.name in ["Line", "Link"]:
                bus_cols = c.static.columns[c.static.columns.str.startswith("bus")]
                used_buses = pd.unique(c.static[bus_cols].values.ravel())
                drop_buses -= set(used_buses)

    else:
        drop_buses = m.buses[~m.buses.country.isin(countries)].index

    m.remove("Bus", list(drop_buses))

    return m


def drop_country(
    n: pypsa.Network,
    countries: list[str],
    keep: list[str] = [],
) -> pypsa.Network:
    """
    Remove selected countries from the network.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to trim.
    countries : list[str]
        List of countries or subregions to remove.
    keep : list[str], default []
        Index labels of components or buses to keep even if they are in the
        removed countries.

    Returns
    -------
    pypsa.Network
        A copy of the network with the selected countries removed.
    """
    m = n.copy()

    drop_buses = m.buses[m.buses.country.isin(countries)].index
    drop_buses = drop_buses.difference(keep)

    # Remove all components connected to those buses
    for c in m.components:
        bus_cols = c.static.columns[c.static.columns.str.startswith("bus")]
        if len(bus_cols) == 0:
            continue

        touches_drop = c.static[bus_cols].isin(drop_buses).any(axis=1)
        protected = c.static.index.isin(keep)
        mask = touches_drop & ~protected
        if mask.any():
            m.remove(c.name, c.static.index[mask])

    # Finally remove the buses themselves
    m.remove("Bus", drop_buses)

    return m


def get_international_connection(
    n: pypsa.Network,
    country: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract international transmission connections for a given country.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to inspect.
    country : str
        Country code to filter international connections.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        - DataFrame with international line and link connections.
        - Series with aggregated nominal power across the foreign country.
    """
    # Buses in the selected country
    buses = n.buses.index[n.buses.country == country]

    df = pd.DataFrame()

    for c in n.components[["Line", "Link"]]:
        if c.name == "Line":
            attr = "s"
            carrier = "AC"
        else:
            attr = "p"
            carrier = "DC"

        mask0 = c.static.bus0.isin(buses)
        mask1 = c.static.bus1.isin(buses)
        mask = (c.static.carrier == carrier) & (mask0 ^ mask1)

        out = c.static.loc[mask].copy()

        # Pick the "other" bus (the one NOT in the country)
        other_bus = out.bus0.where(~mask0[mask], out.bus1)

        # Map to country via n.buses
        out["comp"] = c.name

        # Aligning differences between lines, uni- and bidirection linkss
        if c.name == "Line":
            out["pu"] = out["s_max_pu"]
        else:
            out["pu"] = (out["p_max_pu"] - out["p_min_pu"]) / 2

        out["nom"] = out[f"{attr}_nom"]
        out["country"] = other_bus.map(n.buses.country)
        out["nom_pu"] = out["nom"] * out["pu"]

        df = pd.concat([df, out[["comp", "country", "nom", "nom_pu"]]])

    w = df.groupby("country")["nom_pu"].sum()

    return df, w


def adjust_international_connection(
    n_spatial: pypsa.Network,
    n_values: pypsa.Network,
) -> pypsa.Network:
    """
    Scale spatial network international connections to match aggregated values.

    Parameters
    ----------
    n_spatial : pypsa.Network
        The spatially resolved network whose capacities should be adjusted.
    n_values : pypsa.Network
        The aggregated reference network used for scaling.

    Returns
    -------
    pypsa.Network
        The adjusted spatial network.
    """
    _, w_values = get_international_connection(n_values, "NL")
    df, w_spatial = get_international_connection(n_spatial, "NL")

    country_w = (w_values / w_spatial).dropna()
    for country, w in country_w.items():
        if w == 1:
            continue

        link_idx = df[(df.comp == "Link") & (df.country == country)].index
        n_spatial.links.loc[link_idx, ["p_nom", "p_nom_max"]] *= w

        line_idx = df[(df.comp == "Line") & (df.country == country)].index
        n_spatial.lines.loc[line_idx, ["s_nom", "s_nom_max"]] *= w

    return n_spatial


def adjust_tennet_connection(
    n: pypsa.Network,
    tennet_capacity: dict,
) -> pypsa.Network:
    """
    Adjust AC transmission line capacities in the Dutch network based on
    specified TenneT connection capacities.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network containing transmission lines and buses.
    tennet_capacity : dict
        Dictionary mapping bus-pair connections to transmission capacities
        in GW. Keys must be formatted as `"bus0-bus1"` strings, independent
        of direction.

    Returns
    -------
    pypsa.Network
        The modified network with updated line capacities and undefined connections removed.
    """
    tennet_set_capacity = {
        frozenset(k.split("-")): v * 1000 for k, v in tennet_capacity.items()
    }

    buses = n.buses.index[n.buses.country == "NL"]

    mask0 = n.lines.bus0.isin(buses)
    mask1 = n.lines.bus1.isin(buses)

    df = n.lines[(n.lines.carrier == "AC") & mask0 & mask1].copy()

    # create row-wise frozenset keys
    df["bus_set"] = [frozenset((b0, b1)) for b0, b1 in zip(df["bus0"], df["bus1"])]

    # Remove networks not defined in tennet_capacity
    missing = [
        index
        for index in df.index
        if df.loc[index, "bus_set"] not in tennet_set_capacity
    ]
    if missing:
        print(f"Dropping line: {missing} ")
        df = df.drop(missing)
        n.remove("Line", missing)

    # map capacities
    n.lines.loc[df.index, "s_nom"] = [
        tennet_set_capacity[bus_set] for bus_set in df["bus_set"]
    ]

    return n


def readjust_offshore_buses(
    n: pypsa.Network,
    nl: pypsa.Network,
) -> pypsa.Network:
    """
    Readjust offshore buses and links between the spatial and NL networks.

    Parameters
    ----------
    n : pypsa.Network
        The original spatial network containing offshore buses.
    nl : pypsa.Network
        The NL network to merge and adjust with offshore links.

    Returns
    -------
    pypsa.Network
        The modified spatial network.
    """
    H2_pipeline_GB = nl.links.loc[
        (nl.links.bus0 == "GBAC H2") & (nl.links.bus1 == "NL10AC H2")
    ].index[0]

    n.links.loc["NLOH001-NL00-Offshore DC", "bus1"] = nl.links.loc[
        "relation/14126301-450-DC", "bus1"
    ]
    n.links.loc["Offshore H2 pipeline NLOH001 H2 -> NL H2", "bus1"] = nl.links.loc[
        H2_pipeline_GB, "bus1"
    ]
    nl.remove("Link", ["relation/14126301-450-DC", H2_pipeline_GB])

    return nl


def readjust_load(
    n_spatial: pypsa.Network,
    n_values: pypsa.Network,
    carriers: list[str] = [],
) -> pypsa.Network:
    """
    Scale spatial load time series to match aggregated load values.

    Parameters
    ----------
    n_spatial : pypsa.Network
        Spatial network with load profiles to adjust.
    n_values : pypsa.Network
        Aggregated network providing target load values.
    carriers : list[str], default []
        Load carriers to adjust.

    Returns
    -------
    pypsa.Network
        The adjusted spatial network.
    """
    weightings = n_spatial.snapshot_weightings.objective

    for carrier in carriers:
        spatial_i = n_spatial.loads[n_spatial.loads.carrier == carrier].index
        total_elec_spatial = (weightings @ n_spatial.loads_t.p_set[spatial_i]).sum()

        values_i = n_values.loads[n_values.loads.carrier == carrier].index
        total_elec_values = (weightings @ n_values.loads_t.p_set[values_i]).sum()

        n_spatial.loads_t.p_set[spatial_i] *= total_elec_values / total_elec_spatial

    return n_spatial


def readjust_renewables(
    n_spatial: pypsa.Network,
    n_values: pypsa.Network,
    mapping: dict[str, str],
) -> pypsa.Network:
    """
    Redistribute aggregated renewable generators from `n_values` across the spatial layout of `n_spatial`.

    Parameters
    ----------
    n_spatial : pypsa.Network
        Spatial network providing generator locations, capacities, and profiles.
    n_values : pypsa.Network
        Aggregated network providing renewable capacities and time series.
    mapping : dict[str, str]
        Mapping from aggregated carrier names to spatial carrier names.

    Returns
    -------
    pypsa.Network
        Copy of `n_spatial` with redistributed renewable generators from `n_values`.
    """

    m = n_spatial.copy()

    # drop all original generators
    carrier_drop = [c for c in mapping.values() if c]
    index_drop = m.generators[m.generators.carrier.isin(carrier_drop)].index

    m.remove("Generator", index_drop)

    for tyndp_c, pypsa_c in mapping.items():
        df_values = n_values.generators[n_values.generators.carrier == tyndp_c]
        df_values_p = n_values.generators_t.p_max_pu[df_values.index]

        df_spatial = n_spatial.generators[n_spatial.generators.carrier == pypsa_c][
            ["bus", "p_nom_max"]
        ].copy()
        df_spatial_p = n_spatial.generators_t.p_max_pu[df_spatial.index]

        if df_values.empty or df_spatial.empty:
            continue

        # compute weights
        df_spatial["weight"] = df_spatial["p_nom_max"] / df_spatial["p_nom_max"].sum()

        # align index naming
        df_spatial.index = df_spatial.index.str.replace(pypsa_c, tyndp_c, regex=False)
        df_spatial_p.columns = df_spatial_p.columns.str.replace(
            pypsa_c, tyndp_c, regex=False
        )

        # --- expansion ---
        len_values = len(df_values)
        len_spatial = len(df_spatial)

        df = df_values.loc[df_values.index.repeat(len_spatial)].copy()

        # assign new index (pypsa index repeated per tyndp row)
        df.index = np.tile(df_spatial.index, len_values)

        # attach weights
        df["weight"] = np.tile(df_spatial["weight"].values, len_values)
        num_cols = df.columns.intersection(WEIGHTING_COLS)
        df[num_cols] = df[num_cols].mul(df["weight"], axis=0)
        df = df.drop(columns="weight")

        # replace default bus with the original
        df["bus"] = df_spatial["bus"]

        weightings = n_values.snapshot_weightings.generators
        p_max_pu_weight = (weightings @ df_values_p * df_values["p_nom"]).sum() / (
            weightings @ df_spatial_p * df["p_nom"]
        ).sum()
        p_max_pu = df_spatial_p * p_max_pu_weight

        m.add("Generator", df.index, **df)
        m.generators_t.p_max_pu[df.index] = p_max_pu

    return m


def readjust_conventionals(
    n_spatial: pypsa.Network,
    n_values: pypsa.Network,
    mapping: dict[str, str],
) -> pypsa.Network:
    """
    Redistribute aggregated conventional links from `n_values` across the spatial layout of `n_spatial`.

    Parameters
    ----------
    n_spatial : pypsa.Network
        Spatial network providing generator locations and capacities.
    n_values : pypsa.Network
        Aggregated network providing conventional capacities and time series.
    mapping : dict[str, str]
        Mapping from aggregated carrier names to spatial carrier names.

    Returns
    -------
    pypsa.Network
        Copy of `n_spatial` with redistributed conventional links from `n_values`.
    """

    m = n_spatial.copy()

    # drop all original links
    carrier_drop = [c for c in mapping.values() if c]
    index_drop = m.links[m.links.carrier.isin(carrier_drop)].index

    m.remove("Link", index_drop)

    for tyndp_c, pypsa_c in mapping.items():
        df_values = n_values.links[n_values.links.carrier == tyndp_c]

        df_spatial = n_spatial.links[n_spatial.links.carrier == pypsa_c][
            ["bus1", "p_nom"]
        ].copy()

        if df_values.empty or df_spatial.empty:
            continue

        # compute weights
        df_spatial["weight"] = df_spatial["p_nom"] / df_spatial["p_nom"].sum()

        # align index naming
        df_spatial.index = df_spatial.index.str.replace(pypsa_c, tyndp_c, regex=False)

        # --- expansion ---
        len_values = len(df_values)
        len_spatial = len(df_spatial)

        df = df_values.loc[df_values.index.repeat(len_spatial)].copy()

        # assign new index (pypsa index repeated per tyndp row)
        df.index = np.tile(df_spatial.index, len_values)

        # attach weights (correct alignment!)
        df["weight"] = np.tile(df_spatial["weight"].values, len_values)
        num_cols = df.columns.intersection(WEIGHTING_COLS)
        df[num_cols] = df[num_cols].mul(df["weight"], axis=0)
        df = df.drop(columns="weight")

        # replace default bus with the original
        if tyndp_c == "h2-ccgt":
            df["bus0"] = df_spatial["bus1"] + df["bus0"].str[2:]
        df["bus1"] = df_spatial["bus1"]

        m.add("Link", df.index, **df)

    return m


def retrieve_electricity_weighting(n):
    """Extract the weightings of electricity demand"""
    df_elec = n.loads[n.loads.carrier == "electricity"].copy()
    df_elec.p_set = n.snapshot_weightings.objective @ n.loads_t.p_set[df_elec.index]
    df_elec.index = df_elec.bus.map(n.buses.location)

    return df_elec.p_set / df_elec.p_set.sum()


def readjust_storages(
    n_spatial: pypsa.Network,
    n_values: pypsa.Network,
    mapping: dict[str, str],
) -> pypsa.Network:
    """
    Redistribute aggregated storages from `n_values` across the spatial layout of `n_spatial`.

    Parameters
    ----------
    n_spatial : pypsa.Network
        Spatial network providing storage locations and capacities.
    n_values : pypsa.Network
        Aggregated network providing storage capacities and time series.
    mapping : dict[str, str]
        Mapping from aggregated carrier names to spatial carrier names.

    Returns
    -------
    pypsa.Network
        Copy of `n_spatial` with redistributed storage from `n_values`.
    """
    m = n_spatial.copy()

    # drop all original stores
    carrier_drop = [c for c in mapping.values() if c]
    index_drop = m.stores[m.stores.carrier.isin(carrier_drop)].index

    m.remove("Store", index_drop)

    for tyndp_c, pypsa_c in mapping.items():
        df_values = n_values.stores[n_values.stores.carrier == tyndp_c]
        df_spatial = n_spatial.stores[n_spatial.stores.carrier == pypsa_c].copy()

        if df_values.empty or df_spatial.empty:
            continue

        # compute weights
        weight = df_spatial["e_nom_max"].replace(np.inf, 0)

        if weight.sum() == 0:
            # If e_nom_max is not defined, use electricity demand as a proxy
            weightings = retrieve_electricity_weighting(n_spatial)
            weight = df_spatial.bus.map(n_spatial.buses.location).map(weightings)

        df_spatial["weight"] = weight / weight.sum()

        # align index naming
        df_spatial.index = df_spatial.index.str.replace(pypsa_c, tyndp_c, regex=False)

        # --- expansion ---
        len_values = len(df_values)
        len_spatial = len(df_spatial)

        df = df_values.loc[df_values.index.repeat(len_spatial)].copy()

        # assign new index (pypsa index repeated per tyndp row)
        df.index = np.tile(df_spatial.index, len_values)

        # attach weights (correct alignment!)
        df["weight"] = np.tile(df_spatial["weight"].values, len_values)
        num_cols = df.columns.intersection(WEIGHTING_COLS)
        df[num_cols] = df[num_cols].mul(df["weight"], axis=0)
        df = df.drop(columns="weight")

        # replace default bus with the original
        df["bus"] = df_spatial["bus"]

        m.add("Store", df.index, **df)

    return m


def attach_h2_exogenous_demand(
    n_spatial: pypsa.Network,
    n_values: pypsa.Network,
) -> pypsa.Network:
    """
    Attach exogenous hydrogen demand to the spatial network.

    The demand is distributed across spatial hydrogen buses according to the
    electricity-demand-derived weightings of the spatial network.

    Parameters
    ----------
    n_spatial : pypsa.Network
        Spatial network to receive the H2 demand.
    n_values : pypsa.Network
        Aggregated network providing H2 exogenous demand profiles.

    Returns
    -------
    pypsa.Network
        The spatial network with added H2 exogenous demand.
    """
    # Extract the weightings of electricity demand to distribute H2 demand
    weightings = retrieve_electricity_weighting(n_spatial)

    # Multiply H2 demand from n_values
    df_spatial = n_spatial.buses[n_spatial.buses.carrier == "H2"]
    df = n_values.loads[n_values.loads.carrier == "H2 exogenous demand"]
    df = df.loc[df.index.repeat(len(df_spatial))].copy()
    df.bus = df_spatial.index

    # Set the index to location to be aligned with the weightings
    p_set = n_values.loads_t.p_set[df.index]
    df.index = df.bus.map(df_spatial.location)
    p_set.columns = df.index
    p_set *= weightings

    # Add the suffix "H2 exogenous demand"
    df.index += " " + "H2 exogenous demand"
    p_set.columns += " " + "H2 exogenous demand"

    n_spatial.add("Load", df.index, **df)
    n_spatial.loads_t.p_set[df.index] = p_set

    return n_spatial


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "final_adjustment_myopic",
            configfiles="config/config.tyndp-isie.yaml",
            clusters="all",
            opts="",
            sector_opts="",
            planning_horizons=2050,
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)
    nl = pypsa.Network(snakemake.input.network_nl)

    n_ext = keep_country(n, ["NL"], include_neighbours=True)
    nl = adjust_international_connection(nl, n_ext)

    tennet_capacity = snakemake.params.tennet_capacity
    if tennet_capacity:
        nl = adjust_tennet_connection(nl, tennet_capacity)

    nl = keep_country(nl, ["NL"])
    n_int = keep_country(n, ["NL"])

    # Ad Hoc lines adjustment in the UK interconnections due to offshore buses
    if snakemake.params.offshore_buses:
        keep = [
            "NLOH001",
            "NLOH001 H2",
            "NLOH001-NL00-Offshore DC",
            "Offshore H2 pipeline NLOH001 H2 -> NL H2",
        ]

        n = drop_country(n, ["NL"], keep=keep)
        nl = readjust_offshore_buses(n, nl)
    else:
        n = drop_country(n, ["NL"])

    # Readjust name scheme in NL model
    replacements = {
        "BEAC H2": "BE H2",
        "BEAC": "BE00",
        "DEAC H2": "DE H2",
        "DEAC": "DE00",
        "DKAC H2": "DK H2",
        "DKAC": "DKW1",
        "GBAC H2": "GB H2",
        "GBAC": "GB00",
        "NOAC H2": "NO H2",
        "NOAC": "NOS0"
    }

    for c in nl.components[["Link", "Line"]]:
        for bus in ["bus0","bus1"]:

            c.static[bus] = c.static[bus].str.replace(
                "|".join(replacements),
                lambda m: replacements[m.group()],
                regex=True,
            )

    # Adjust and distribute TYNDP components but keep the values consistent
    nl = readjust_load(nl, n_int, carriers=["electricity"])
    nl = readjust_renewables(nl, n_int, mapping=snakemake.params.res_tyndp_mapping)
    nl = readjust_conventionals(nl, n_int, mapping=snakemake.params.conv_tyndp_mapping)
    nl = readjust_storages(nl, n_int, mapping=snakemake.params.store_tyndp_mapping)
    nl = attach_h2_exogenous_demand(nl, n_int)

    # Drop NL carrier components to prevent merging collition
    nl.remove("Carrier", nl.carriers.index.intersection(n.carriers.index))

    # Drop NL global constraints
    nl.remove("GlobalConstraint", nl.global_constraints.index)

    m = n.merge(nl)

    # Postmerge adjustement to avoid warnings
    m.loads_t.p_set.fillna(m.loads.p_set, inplace=True)
    m.lines.drop(["r", "x", "b"], axis=1, inplace=True)

    m.export_to_netcdf(snakemake.output.network)
