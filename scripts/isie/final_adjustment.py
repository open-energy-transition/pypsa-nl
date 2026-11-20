# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-FileCopyrightText: Open Energy Transition gGmbH
#
# SPDX-License-Identifier: MIT
import logging
from pathlib import Path

import pandas as pd
import pypsa

from scripts._helpers import (
    configure_logging,
    load_costs,
    set_scenario_config,
)
from scripts.isie._helpers import (
    disable_config,
    make_all_battery_extendable,
    make_renewables_expandable,
)
from scripts.isie.final_adjustment_regionalize import readjust_tennet_storages


def add_home_battery(n, costs, countries):

    nodes = n.buses.index[(n.buses.carrier == "AC") & n.buses.country.isin(countries)]

    n.add("Carrier", "home battery")

    n.add(
        "Bus",
        nodes + " home battery",
        location=nodes,
        carrier="home battery",
        unit="MWh_el",
    )

    n.add(
        "Store",
        nodes + " home battery",
        bus=nodes,
        location=nodes,
        e_cyclic=True,
        e_nom_extendable=True,
        carrier="home battery",
        capital_cost=costs.at["home battery storage", "capital_cost"],
        lifetime=costs.at["battery storage", "lifetime"],
    )

    n.add(
        "Link",
        nodes + f" home battery charger-{year}",
        bus0=nodes,
        bus1=nodes + " home battery",
        carrier="home battery charger",
        efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
        capital_cost=costs.at["home battery inverter", "capital_cost"],
        p_nom_extendable=True,
        lifetime=costs.at["battery inverter", "lifetime"],
    )

    n.add(
        "Link",
        nodes + f" home battery discharger-{year}",
        bus0=nodes + " home battery",
        bus1=nodes,
        carrier="home battery discharger",
        efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
        marginal_cost=costs.at["home battery storage", "marginal_cost"],
        p_nom_extendable=True,
        lifetime=costs.at["battery inverter", "lifetime"],
    )

    return n


def readjust_tennet_load(
    n: pypsa.Network,
    tennet_xlsx: str,
    refactor: dict | bool = False,
    scenario: str = "KM",
    year: int = 2050,
) -> pypsa.Network:

    df_profile = pd.read_excel(
        tennet_xlsx,
        sheet_name="Hourly demand profiles",
        header=[0, 1, 2, 3],
        index_col=0,
    )

    df_profile = df_profile[scenario][year]
    df_profile.columns = df_profile.columns.map("_".join)

    hours = (n.snapshots - n.snapshots[0]).total_seconds() / 3600 + 1
    df_profile.index = df_profile.index.map(dict(zip(hours.astype(int), n.snapshots)))
    df_profile = df_profile.loc[df_profile.index.dropna()]

    if refactor:
        for k, v in refactor.items():
            df_profile[k] *= v

    to_change = n.loads[
        (n.loads["carrier"] == "electricity")
        & (n.loads["bus"].map(n.buses.country) == "NL")
    ].index[0]

    n.loads_t.p_set[to_change] = -df_profile.T.sum()

    return n


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "final_adjustment",
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

    costs = load_costs(snakemake.input.tech_costs)
    year = int(snakemake.wildcards.horizon)
    n = pypsa.Network(snakemake.input.network)

    if tennet_adjustment["load"]["enable"]:
        n = readjust_tennet_load(
            n, tennet_xlsx, refactor=tennet_adjustment["load"]["refactor"], year=year
        )

    battery_system = tennet_adjustment["battery_system"]
    if battery_system["TSO"]["enable"]:
        n = readjust_tennet_storages(
            n,
            tennet_xlsx,
            carrier="battery",
            battery_system=battery_system["TSO"],
            year=year,
            regionalized=False,
        )

    if battery_system["DSO"]["enable"]:
        n = add_home_battery(n, costs, countries=["NL"])

        n = readjust_tennet_storages(
            n,
            tennet_xlsx,
            carrier="home battery",
            battery_system=battery_system["DSO"],
            year=year,
            regionalized=False,
        )

    n = make_renewables_expandable(
        n,
        res_carrier=snakemake.config["electricity"]["tyndp_renewable_carriers"],
        min_ext_factor=tennet_adjustment["min_ext_factor"],
    )

    if battery_system["extendable"]:
        n = make_all_battery_extendable(n)

    breakpoint()

    n.export_to_netcdf(snakemake.output.network)
