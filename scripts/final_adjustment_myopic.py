# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-FileCopyrightText: Open Energy Transition gGmbH
#
# SPDX-License-Identifier: MIT
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from scripts._helpers import (
    configure_logging,
    set_scenario_config,
)

WEIGHTING_COLS = [
    "p_init",
    "p_nom",
    "p_nom_max",
    "p_nom_min",
    "p_nom_mod",
    "p_nom_opt",
    "p_nom_set",
    "p_set",
    "e_initial",
    "e_nom",
    "e_nom_max",
    "e_nom_min",
    "e_nom_mod",
    "e_nom_opt",
    "e_nom_set",
    "e_set",
]

TYNDP_TECH = [
    "nuclear",
    "gas-conv-old1",
    "gas-conv-old2",
    "gas-ccgt-old1",
    "gas-ccgt-old2",
    "gas-ccgt-new",
    "gas-ccgt-ccs",
    "gas-ccgt-present1",
    "gas-ccgt-present2",
    "gas-ocgt-old",
    "gas-ocgt-new",
    "coal-old1",
    "coal-old2",
    "coal-new",
    "coal-ccs",
    "lignite-old1",
    "lignite-old2",
    "lignite-new",
    "lignite-ccs",
    "oil-light",
    "oil-heavy-old1",
    "oil-heavy-old2",
    "oil-shale-old",
    "oil-shale-new",
    "h2-ccgt",
    "other-non-res-gas-ccgt-old1",
    "other-res-biomass",
    "H2 Electrolysis",
    "SMR CC",
]

PYPSA_TECH = [
    "OCGT",
    "CCGT",
    "H2 Electrolysis",
    "SMR CC",
    "SMR",
    "H2 Fuel Cell",
    "H2 turbine",
]


def disable_config(d):
    return {
        k: disable_config(v) if isinstance(v, dict) else False if v is True else v
        for k, v in d.items()
    }


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
        n_spatial.lines.loc[line_idx, "s_max_pu"] = 1.0

    return n_spatial


def adjust_tennet_connection(
    n: pypsa.Network,
    tennet_capacity: str,
    tennet_xlsx: str,
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

    df_cap = pd.read_excel(
        tennet_xlsx,
        sheet_name="Transfer capacities 2040",
        header=[0],
        index_col=0,
    ).filter(like="R", axis=0)
    df_cap.columns = [
        "circuit",
        "summer max",
        "winter max",
        "summer effective",
        "winter effective",
    ]

    buses = n.buses.index[n.buses.country == "NL"]

    mask0 = n.lines.bus0.isin(buses)
    mask1 = n.lines.bus1.isin(buses)

    df = n.lines[(n.lines.carrier == "AC") & mask0 & mask1].copy()

    # create row-wise frozenset keys
    df_cap["bus_set"] = [
        frozenset([f"NL{i[1:3]}AC", f"NL{i[5:7]}AC"]) for i in df_cap.index
    ]
    df["bus_set"] = [frozenset((b0, b1)) for b0, b1 in zip(df["bus0"], df["bus1"])]

    # Remove networks not defined in tennet_capacity
    missing = list(df["bus_set"][~df["bus_set"].isin(df_cap["bus_set"])].index)
    if missing:
        print(f"Dropping line: {missing} ")
        df = df.drop(missing)
        n.remove("Line", missing)

    df = df.reset_index().merge(df_cap, on="bus_set").set_index("name")

    # map capacities
    n.lines.loc[df.index, "s_nom"] = df[tennet_capacity["type"]]

    if tennet_capacity["extendable"]:
        n.lines.loc[df.index, "s_nom_extendable"] = True
        if tennet_capacity["method"] == "circuit":
            n.lines.loc[df.index, "s_nom_max"] = np.maximum(
                tennet_capacity["max_circuit"]
                / df["circuit"]
                * df[tennet_capacity["type"]],
                df[tennet_capacity["type"]],
            )
            log_info = (
                f"extendable by a maximum of {tennet_capacity['max_circuit']} per line"
            )
        elif tennet_capacity["method"] == "s_max_pu":
            n.lines.loc[df.index, "s_nom_max"] = (
                df[tennet_capacity["type"]] * tennet_capacity["s_max_pu"]
            )
            log_info = (
                f"extendable by a factor of {tennet_capacity['s_nom_max']} per line"
            )
        elif tennet_capacity["method"] == "s_nom_max":
            n.lines.loc[df.index, "s_nom_max"] = np.maximum(
                tennet_capacity["s_nom_max"],
                df[tennet_capacity["type"]],
            )
            log_info = (
                f"extendable by a up to {tennet_capacity['s_nom_max']} MW per line"
            )
        else:
            raise ValueError(
                "Transmission network are extendable but no method defined."
            )

    else:
        log_info = "not extendable"

    n.lines.loc[df.index, "s_max_pu"] = 1.0

    logger.info(
        "Readjust transmission:\n"
        f"- Value based on 2040 {tennet_capacity['type']} capacity.\n"
        f"- Set transmission line to {log_info}.\n"
        "- Distributed using TenneT's factor."
    )
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

    # Connect Great Britain H2 pipeline with Rotterdam
    H2_pipeline_GB = nl.links.loc[
        (nl.links.bus0 == "GBAC H2") & (nl.links.bus1 == "NL10AC H2")
    ].index[0]
    n.links.loc["Offshore H2 pipeline NLOH001 H2 -> NL H2", "bus1"] = nl.links.loc[
        H2_pipeline_GB, "bus1"
    ]
    # Connect offshore DC lines of the Netherlands
    n.links.loc["NLOH001-NL00-Offshore DC", "bus1"] = nl.links.loc[
        "relation/14126301-450-DC", "bus1"
    ]
    # Remove duplicate links
    nl.remove("Link", ["relation/14126301-450-DC", H2_pipeline_GB])

    # Set import destination for Ammonia to Rotterdam (NL10AC H2) and Vlissingen (NL15AC H2)
    H2_import = ["H2 import Ammonia -> NL - low", "H2 import Ammonia -> NL - high"]
    H2_import_bus = ["NL10AC H2", "NL15AC H2"]
    df_import = n.links.loc[np.tile(H2_import, len(H2_import_bus))].copy()
    df_import["bus1"] = pd.Series(H2_import_bus).repeat(len(H2_import)).values
    df_import.index = [
        idx.replace("NL", bus1)
        for idx, bus1 in zip(df_import.index, df_import["bus1"].str.replace(" H2", ""))
    ]
    # Remove the single import link and replace them with two links.
    n.remove("Link", H2_import)
    n.add("Link", df_import.index, **df_import)

    return n, nl


def extract_regionalization(
    tennet_xlsx: str,
    scenario: str = "KM",
    year: int = 2050,
    filter: dict = {"CARRIER": ["Electricity"]},
    merge_level: bool = False,
):

    df_region = pd.read_excel(
        tennet_xlsx,
        sheet_name="Regionalisation keys 16 regions",
        header=[0, 1, 2, 3, 4, 5, 6],
        index_col=0,
    )

    df_region = df_region.loc[~df_region.index.duplicated(keep="first")]
    df_region = df_region[scenario][year]

    for key, values in filter.items():
        df_region = df_region.loc[
            :, df_region.columns.get_level_values(key).isin(values)
        ]

    keep = ["CARRIER", "LEVEL", "TYPE", "SECTOR"]
    drop = [x for x in df_region.columns.names if x not in keep]

    df_region.columns = df_region.columns.droplevel(drop)
    df_region = df_region.drop("Total").dropna()
    df_region["bus"] = [str(i).split("_")[0] for i in df_region.index]
    df_region["bus"] = df_region["bus"].apply(lambda x: f"0{x}" if len(x) == 1 else x)
    df_region = df_region.set_index("bus")

    if not merge_level:
        keep = ["LEVEL", "SECTOR"]
        drop = [x for x in df_region.columns.names if x not in keep]

        df_region.columns = df_region.columns.droplevel(drop).map("_".join)
        return df_region

    df_split = pd.read_excel(
        tennet_xlsx,
        sheet_name="TSO_DSO_split",
        header=[0],
        index_col=[0, 1, 2],
    )
    s_split = df_split[f"{scenario}_{year}"]

    factors = df_region.columns.map(
        lambda c: (
            (1 - s_split[c[0]][c[2]][c[3]])
            if c[1] == "DSO"
            else s_split[c[0]][c[2]][c[3]]
        )
    )

    df_region_adj = df_region.mul(factors, axis="columns")
    df_region_adj = df_region_adj.T.groupby("SECTOR").sum().T

    return df_region_adj


def readjust_tennet_load(
    n: pypsa.Network, tennet_xlsx: str, scenario: str = "KM", year: int = 2050
):

    to_drop = n.loads[
        (n.loads["carrier"] == "electricity")
        & (nl.loads["bus"].map(nl.buses.country) == "NL")
    ].index

    n.remove("Load", to_drop)

    df_region = extract_regionalization(
        tennet_xlsx,
        scenario=scenario,
        year=year,
        filter={"CARRIER": ["Electricity"], "TYPE": ["Demand"]},
        merge_level=False,
    )

    df_profile = pd.read_excel(
        tennet_xlsx,
        sheet_name="Hourly demand profiles",
        header=[0, 1, 2, 3],
        index_col=0,
    )

    df_profile = df_profile[scenario][year]
    df_profile.columns = df_profile.columns.map("_".join)

    hours = (n.snapshots - n.snapshots[0]).total_seconds() / 3600 + 1
    df_profile.index = df_profile.index.map(dict(zip(hours.astype(int), nl.snapshots)))
    df_profile = df_profile.loc[df_profile.index.dropna()]

    for operator in ["TSO", "DSO"]:
        df1 = df_region.filter(like=operator).copy()
        df2 = df_profile.filter(like=operator).copy()

        columns = df1.columns.intersection(df2.columns)
        columns_missing = df1.columns.difference(df2.columns)

        logger.info(f"Missing columns: {columns_missing}")

        df1 = df1[columns]
        df2 = df2[columns]

        df_dynamic = -df2 @ df1.T
        df_dynamic = df_dynamic.loc[:, df_dynamic.sum() > 0]

        voltage = " low voltage" if operator == "DSO" else ""
        bus = df_dynamic.columns.astype(str)
        df_static = pd.DataFrame(
            {
                "bus": "NL" + bus + "AC" + voltage,
                "carrier": "electricity",
            },
            index="NL" + bus + "AC " + operator,
        )
        df_dynamic.columns = "NL" + bus + "AC " + operator

        n.add(
            "Load",
            df_static.index,
            bus=df_static["bus"],
            carrier=df_static["carrier"],
            p_set=df_dynamic,
        )

    logger.info(
        "Readjust load:\n"
        "- Split NL demand to TSO and DSO,\n"
        "- Distributed using TenneT's factor."
    )

    return n


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


# def make_renewables_expandable(
#     n_spatial: pypsa.Network,
#     n_values: pypsa.Network,
#     mapping: dict[str, str],
#     expandable: list,
# ) -> pypsa.Network:

#     tyndp_p_nom_max = n_spatial.generators.groupby("carrier")["p_nom_max"].sum()

#     m = n_values.copy()

#     # use the res_tyndp_mapping to distribute the p_nom_max. If there is more carrier per p_nom_max, use these to split it
#     gens = m.generators
#     for b_carrier, t_carrier in mapping.items():
#         if t_carrier not in tyndp_p_nom_max.index or b_carrier not in expandable:
#             continue
#         idx = gens.index[gens.carrier == b_carrier]
#         if len(idx) == 0:
#             continue
#         # split proportionally to existing p_nom_max (if finite), else p_nom, else equally
#         weights = gens.loc[idx, "p_nom_max"]
#         if not (
#             weights.replace([float("inf")], float("nan")).notna().all()
#             and weights.sum() > 0
#         ):
#             weights = gens.loc[idx, "p_nom"]
#         if weights.sum() <= 0:
#             weights = pd.Series(1.0, index=idx)
#         gens.loc[idx, "p_nom_max"] = np.maximum(
#             tyndp_p_nom_max[t_carrier] * weights / weights.sum(), gens.loc[idx, "p_nom"]
#         )

#     exp_idx = gens.index[gens.carrier.isin(expandable)]
#     gens.loc[exp_idx, "p_nom_extendable"] = True

#     return m


def set_renewable_p_nom_max(
    n: pypsa.Network,
    expandable: dict,
) -> pypsa.Network:

    gens = n.generators
    for carrier, value in expandable.items():
        idx = gens.index[gens.carrier == carrier]
        if value:
            gens.loc[idx, "p_nom_max"] = value

    return n


def make_renewables_expandable(
    n: pypsa.Network,
    expandable: dict,
    countries: list = ["NL"],
) -> pypsa.Network:

    gens = n.generators
    buses_idx = n.buses.index[n.buses.country.isin(countries)]
    exp_idx = gens.index[
        gens.carrier.isin(expandable.keys()) & gens.bus.isin(buses_idx)
    ]
    gens.loc[exp_idx, "p_nom_extendable"] = True

    return n


def readjust_tennet_renewables(
    n_spatial: pypsa.Network,
    n_values: pypsa.Network,
    tennet_xlsx: str,
    mapping_tyndp: dict[str, str],
    mapping_tennet: dict[str, str],
    scenario: str = "KM",
    year: int = 2050,
):

    df_region = extract_regionalization(
        tennet_xlsx,
        scenario=scenario,
        year=year,
        filter={"TYPE": ["Supply"]},
        merge_level=True,
    )

    m = n_spatial.copy()

    # drop all original generators
    carrier_drop = [c for c in mapping_tyndp.values() if c]
    index_drop = m.generators[m.generators.carrier.isin(carrier_drop)].index

    m.remove("Generator", index_drop)

    for tyndp_c, tennet_c in mapping_tennet.items():
        df_values = n_values.generators[n_values.generators.carrier == tyndp_c]
        df_values_p = n_values.generators_t.p_max_pu[df_values.index]

        factor = df_region[tennet_c]

        # --- expansion ---
        len_values = len(df_values)
        len_spatial = len(factor)

        df = df_values.loc[df_values.index.repeat(len_spatial)].copy()

        df["weight"] = np.tile(factor, len_values)
        num_cols = df.columns.intersection(WEIGHTING_COLS)
        df[num_cols] = df[num_cols].mul(df["weight"], axis=0)
        df["bus"] = [f"NL{i}AC" for i in factor.index]
        df.index = [f"NL{i}AC 0 {tyndp_c}-{year}" for i in factor.index]

        # Set the p_max_pu the same throughout the region.
        df_spatial_p = df_values_p[df_values.index.repeat(len_spatial)].copy()
        df_spatial_p.columns = [f"NL{i}AC 0 {tyndp_c}-{year}" for i in factor.index]

        # Filter generators with 0 p_nom
        df = df[df["p_nom"] > 0]
        p_max_pu = df_spatial_p[df.index]

        m.add("Generator", df.index, **df)
        m.generators_t.p_max_pu[df.index] = p_max_pu

    logger.info(
        "Readjust renewables:\n"
        "- Total renewables capacity in NL to be the same as Open-TYNDP,\n"
        "- Distributed using TenneT's factor."
    )

    return m


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

    logger.info(
        "Readjust renewables:\n"
        "- Total renewables capacity in NL to be the same as Open-TYNDP,\n"
        "- Distributed using PyPSA-Eur electricity load demand."
    )

    return m


def distribute_conventionals(
    n_spatial: pypsa.Network,
    n_values: pypsa.Network,
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
    carrier_drop = [c for c in PYPSA_TECH if c]
    index_drop = m.links[m.links.carrier.isin(carrier_drop)].index

    m.remove("Link", index_drop)

    df_all = pd.DataFrame()

    for carrier in TYNDP_TECH:
        df_values = n_values.links[n_values.links.carrier == carrier]
        if df_values.empty:
            continue

        year = df_values["build_year"].max()

        bus_index = n_spatial.buses[n_spatial.buses.carrier == "AC"].index
        name_index = [f"{bus} {carrier}-{year}" for bus in bus_index]
        df_spatial = pd.DataFrame({"bus": bus_index, "p_nom": 0.0}, index=name_index)

        len_values = len(df_values)
        len_spatial = len(name_index)

        df = df_values.loc[df_values.index.repeat(len_spatial)].copy()
        df_spatial = df_spatial.loc[np.tile(name_index, len_values)].copy()

        # assign new index (pypsa index repeated per tyndp row)
        if len_values > 1:
            for v in range(len_values):
                s_tyndp_sub = df_values.iloc[v]
                name = s_tyndp_sub.name
                name = name.replace(f"{s_tyndp_sub.bus1}-", "").replace(
                    f"{s_tyndp_sub.build_year}-", ""
                )
                range_index = [s + len_spatial * v for s in range(len_spatial)]

                new_index = df_spatial.index.to_numpy(copy=True)
                new_index[range_index] = [
                    x.replace(carrier, name) for x in new_index[range_index]
                ]
                df_spatial.index = new_index

        # use p_nom_group to distribute capacity of TYNDP technologies
        df["p_nom_group"] = df["p_nom"]
        df["p_nom"] = 0
        df["p_nom_min"] = 0
        df["p_nom_extendable"] = True

        df.index = df_spatial.index

        if carrier == "h2-ccgt":
            df["bus0"] = df_spatial["bus"] + " H2"
            df["bus1"] = df_spatial["bus"]
        elif carrier == "H2 Electrolysis":
            df["bus0"] = df_spatial["bus"]
            df["bus1"] = df_spatial["bus"] + " H2"
        else:
            df["bus1"] = df_spatial["bus"]

        df_all = pd.concat([df_all, df])

    m.add("Link", df_all.index, **df_all)

    logger.info(
        "Readjust conventionals:\n"
        "- Total conventionals capacity in NL to be the same as Open-TYNDP,\n"
        f"- Distributed by the following carriers:{'\n-'.join(df_all.carrier.unique())}"
    )

    return m


def readjust_tennet_storages(n, tennet_xlsx, battery_system, scenario="KM", year=2050):

    factor = extract_regionalization(
        tennet_xlsx,
        scenario=scenario,
        year=year,
        filter={"SECTOR": ["Battery_system"]},
        merge_level=True,
    )["Battery_system"]

    index = factor.index
    df_sys = pd.DataFrame(
        {
            "factor": factor,
            "store_name": [f"NL{i}AC battery-{year}" for i in index],
            "charger_name": [f"NL{i}AC battery charger-{year}" for i in index],
            "discharger_name": [f"NL{i}AC battery discharger-{year}" for i in index],
            "power_capacity": factor * battery_system["power_cap"],
            "energy_capacity": factor
            * battery_system["power_cap"]
            * battery_system["max_load"],
        }
    )

    n.links.loc[df_sys["charger_name"], "p_nom_extendable"] = battery_system[
        "extendable"
    ]
    n.links.loc[df_sys["charger_name"], "p_nom"] = [i for i in df_sys["power_capacity"]]

    n.links.loc[df_sys["discharger_name"], "p_nom_extendable"] = battery_system[
        "extendable"
    ]
    n.links.loc[df_sys["discharger_name"], "p_nom"] = [
        i for i in df_sys["power_capacity"]
    ]

    n.stores.loc[df_sys["store_name"], "e_nom_extendable"] = battery_system[
        "extendable"
    ]
    n.stores.loc[df_sys["store_name"], "e_nom"] = [i for i in df_sys["energy_capacity"]]

    logger.info(
        "Readjust storage:\n"
        f"- Set battery to {battery_system['power_cap']} MW,\n"
        "- All located in high voltage.\n"
        "- Distributed using TenneT's factor."
    )

    return n


def retrieve_electricity_weighting(n):
    """Extract the weightings of electricity demand"""
    df_elec = n.loads[(n.loads.carrier == "electricity")].copy()
    df_elec.p_set = n.snapshot_weightings.objective @ n.loads_t.p_set[df_elec.index]
    df_elec.index = df_elec.bus.map(n.buses.location)
    df_elec = df_elec.groupby(df_elec.index)["p_set"].sum()

    return df_elec / df_elec.sum()


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
    log_map = ""

    for tyndp_c, pypsa_c in mapping.items():
        df_values = n_values.stores[n_values.stores.carrier == tyndp_c]
        df_spatial = n_spatial.stores[n_spatial.stores.carrier == pypsa_c].copy()

        if df_values.empty or df_spatial.empty:
            continue

        # compute weights
        weight = df_spatial["e_nom_max"].replace(np.inf, 0)

        if weight.sum() == 0:
            log_map += f"- {tyndp_c}: electricity\n"

            # If e_nom_max is not defined, use electricity demand as a proxy
            weightings = retrieve_electricity_weighting(n_spatial)
            weight = df_spatial.bus.map(n_spatial.buses.location).map(weightings)
        else:
            log_map += f"- {tyndp_c}: {pypsa_c}\n"

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
        df["bus"] = df_spatial["bus"] + df["carrier"].str[2:]
        df_store = df.copy()

        if tyndp_c == "H2 cavern-storage":
            # Add buses
            df_values = n_values.buses[n_values.buses.carrier == tyndp_c]

            df = df_values.loc[df_values.index.repeat(len_spatial)].copy()
            df.index = df_store.index.str.replace(f"-{year}", "", regex=False)
            df["x"] = n_spatial.buses.loc[df_spatial["bus"], "x"].values
            df["y"] = n_spatial.buses.loc[df_spatial["bus"], "x"].values

            df = df.loc[(df_store["e_nom"] > 0).values]
            m.add("Bus", df.index, **df)

            # Add charger and discharger
            for c in ["charger", "discharger"]:
                df_values = n_values.links[n_values.links.carrier == tyndp_c + f" {c}"]

                df = df_values.loc[df_values.index.repeat(len_spatial)].copy()
                df.index = df_store.index.str.replace(
                    "storage", f"storage {c}", regex=False
                )
                df["bus0"] = (
                    df_store["bus"].values
                    if c == "discharger"
                    else df_spatial["bus"].values
                )
                df["bus1"] = (
                    df_store["bus"].values
                    if c == "charger"
                    else df_spatial["bus"].values
                )

                df["weight"] = np.tile(df_spatial["weight"].values, len_values)
                num_cols = df.columns.intersection(WEIGHTING_COLS)
                df[num_cols] = df[num_cols].mul(df["weight"], axis=0)
                df = df.drop(columns="weight")

                df = df[df["p_nom"] > 0]
                m.add("Link", df.index, **df)

        # Add storage
        df_store = df_store[df_store["e_nom"] > 0]
        m.add("Store", df_store.index, **df_store)

    logger.info(
        "Readjust storages:\n"
        "- Total storages in NL to be the same as Open-TYNDP.\n"
        f"- This effects {mapping.keys()}\n."
        f"- Distributed by the following carriers:\n{log_map}"
    )

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

    logger.info(
        "Readjust h2 exogenous demand:\n"
        "- Total h2 exogenous demand in NL to be the same as Open-TYNDP,\n"
        "- Distributed using PyPSA-Eur electricity load demand."
    )

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

    tennet_adjustment = snakemake.params.tennet_adjustment
    tennet_xlsx = tennet_adjustment["tennet_xlsx"]
    if tennet_xlsx and Path(tennet_xlsx).exists():
        logger.info(f"Retrieving TenneT's data from {tennet_xlsx}")
    else:
        tennet_adjustment = disable_config(tennet_adjustment)
        logger.info("No TenneT's data available, disabling related config.")

    n = pypsa.Network(snakemake.input.network)
    nl = pypsa.Network(snakemake.input.network_nl)

    n_ext = keep_country(n, ["NL"], include_neighbours=True)
    nl = adjust_international_connection(nl, n_ext)

    tennet_capacity = tennet_adjustment["tennet_capacity"]
    if tennet_capacity["enable"]:
        nl = adjust_tennet_connection(nl, tennet_capacity, tennet_xlsx)

    nl = keep_country(nl, ["NL"])
    n_int = keep_country(n, ["NL"])

    # Ad Hoc lines adjustment in the UK interconnections due to offshore buses
    if snakemake.params.offshore_buses:
        keep = [
            "NLOH001",
            "NLOH001 H2",
            "NLOH001-NL00-Offshore DC",
            "Offshore H2 pipeline NLOH001 H2 -> NL H2",
            "H2 import Ammonia -> NL - low",
            "H2 import Ammonia -> NL - high",
        ]

        n = drop_country(n, ["NL"], keep=keep)
        n, nl = readjust_offshore_buses(n, nl)
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
        "NOAC": "NOS0",
    }

    for c in nl.components[["Link", "Line"]]:
        for bus in ["bus0", "bus1"]:
            c.static[bus] = c.static[bus].str.replace(
                "|".join(replacements),
                lambda m: replacements[m.group()],
                regex=True,
            )

    # Adjust and distribute TYNDP components but keep the values consistent
    year = int(snakemake.wildcards.horizon)
    store_tyndp_mapping = snakemake.params.store_tyndp_mapping
    res_tyndp_mapping = snakemake.params.res_tyndp_mapping
    res_tennet_mapping = snakemake.params.res_tennet_mapping

    if tennet_adjustment["load"]:
        nl = readjust_tennet_load(nl, tennet_xlsx, year=year)
    else:
        nl = readjust_load(nl, n_int, carriers=["electricity"])

    battery_system = tennet_adjustment["battery_system"]
    if battery_system["enable"]:
        nl = readjust_tennet_storages(
            nl, tennet_xlsx, battery_system=battery_system, year=year
        )
        store_tyndp_mapping.pop("battery", None)

    renewables = tennet_adjustment["renewables"]
    if renewables["expandable"]:
        n_int = set_renewable_p_nom_max(n_int, expandable=renewables["expandable"])

    if renewables["by_tennet"]:
        nl = readjust_tennet_renewables(
            nl,
            n_int,
            tennet_xlsx,
            mapping_tyndp=res_tyndp_mapping,
            mapping_tennet=res_tennet_mapping,
            year=year,
        )
    else:
        nl = readjust_renewables(nl, n_int, mapping=res_tyndp_mapping)

    nl = distribute_conventionals(nl, n_int)
    nl = readjust_storages(nl, n_int, mapping=store_tyndp_mapping)

    nl = attach_h2_exogenous_demand(nl, n_int)

    # Drop NL carrier components to prevent merging collition
    nl.remove("Carrier", nl.carriers.index.intersection(n.carriers.index))

    # Drop NL global constraints
    nl.remove("GlobalConstraint", nl.global_constraints.index)

    m = n.merge(nl)

    if renewables["expandable"]:
        m = make_renewables_expandable(
            m, expandable=renewables["expandable"], countries=["NL"]
        )

    # Postmerge adjustment to avoid warnings
    m.loads_t.p_set.fillna(m.loads.p_set, inplace=True)
    m.lines.drop(["r", "x", "b"], axis=1, inplace=True)

    m.export_to_netcdf(snakemake.output.network)
