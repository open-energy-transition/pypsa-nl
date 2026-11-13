# SPDX-FileCopyrightText: : 2023- The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT


import pypsa
from snakemake.script import Snakemake


def add_battery_duration_fix(n: pypsa.Network, snakemake: Snakemake) -> None:
    """
    Constrain the battery energy/power ratio for extendable batteries in NL.

    Parameters
    ----------
    n : pypsa.Network
        Network with an already created linopy model (``n.model``).
    snakemake : snakemake.script.Snakemake
        Snakemake object providing access to the workflow ``config``.

    Returns
    -------
    None
        The network's optimisation model is modified in place.
    """

    bus_country = n.buses.index[n.buses.country.isin(["NL"])]

    for carrier in ["battery", "home battery"]:
        level = "DSO" if carrier == "home battery" else "TSO"

        energy_bool = n.stores.carrier.isin([carrier]) & n.stores.bus.isin(bus_country)
        energy_ext = n.stores[energy_bool].query("e_nom_extendable")

        charger_bool = n.links.carrier.isin([f"{carrier} charger"]) & n.links.bus0.isin(
            bus_country
        )
        chargers_ext = n.links[charger_bool].query("p_nom_extendable")

        max_hours = snakemake.config["tennet_adjustment"]["battery_system"][level].get(
            "max_load", snakemake.config["electricity"]["max_hours"]["battery"]
        )

        if not energy_ext.empty and not chargers_ext.empty:
            expr = (
                n.model["Store-e_nom"].loc[energy_ext.index]
                == n.model["Link-p_nom"].loc[chargers_ext.index] * max_hours
            )

            n.model.add_constraints(expr, name=f"NL-{carrier}-duration")


def distribute_TYNDP_carrier_limit(n: pypsa.Network, snakemake: Snakemake) -> None:
    """
    Distribute TYNDP capacity targets over extendable links and generators.

    Components with a non-null ``p_nom_group`` (the TYNDP capacity target for
    their carrier) and ``p_nom_extendable`` set are grouped by carrier.
    Constraints are skipped for empty selections.

    Parameters
    ----------
    n : pypsa.Network
        Network with an already created linopy model (``n.model``).
    snakemake : snakemake.script.Snakemake
        Snakemake object providing access to the workflow ``config``.

    Returns
    -------
    None
        The network's optimisation model is modified in place.
    """
    expandable = snakemake.config["tennet_adjustment"]["extendable_carriers"][
        "Generator"
    ]

    # For Links
    link_p_nom_bool = ~n.links.p_nom_group.isna() & ~n.links.carrier.isin(expandable)
    link_p_nom = n.links[link_p_nom_bool].query("p_nom_extendable")

    if not link_p_nom.empty:
        expr_1 = (
            n.model["Link-p_nom"]
            .loc[link_p_nom.index]
            .groupby(link_p_nom.carrier)
            .sum()
            == link_p_nom.groupby("carrier")["p_nom_group"].mean()
        )

        n.model.add_constraints(expr_1, name="Link-dist-TYNDP")

    link_p_min_bool = ~n.links.p_nom_group.isna() & n.links.carrier.isin(expandable)
    link_p_min = n.links[link_p_min_bool].query("p_nom_extendable")

    if not link_p_min.empty:
        expr_2 = (
            n.model["Link-p_nom"]
            .loc[link_p_min.index]
            .groupby(link_p_min.carrier)
            .sum()
            >= link_p_min.groupby("carrier")["p_nom_group"].mean()
        )

        n.model.add_constraints(expr_2, name="Link-min-TYNDP")

    # For Generators
    gen_p_nom_bool = ~n.generators.p_nom_group.isna() & ~n.generators.carrier.isin(
        expandable
    )
    gen_p_nom = n.generators[gen_p_nom_bool].query("p_nom_extendable")

    if not gen_p_nom.empty:
        expr_3 = (
            n.model["Generator-p_nom"]
            .loc[gen_p_nom.index]
            .groupby(gen_p_nom.carrier)
            .sum()
            == gen_p_nom.groupby("carrier")["p_nom_group"].mean()
        )

        n.model.add_constraints(expr_3, name="Generator-dist-TYNDP")

    gen_p_min_bool = ~n.generators.p_nom_group.isna() & n.generators.carrier.isin(
        expandable
    )
    gen_p_min = n.generators[gen_p_min_bool].query("p_nom_extendable")

    if not gen_p_min.empty:
        expr_4 = (
            n.model["Generator-p_nom"]
            .loc[gen_p_min.index]
            .groupby(gen_p_min.carrier)
            .sum()
            >= gen_p_min.groupby("carrier")["p_nom_group"].mean()
        )

        n.model.add_constraints(expr_4, name="Generator-min-TYNDP")


def custom_extra_functionality(n, snapshots, snakemake):

    add_battery_duration_fix(n, snakemake)

    distribute_TYNDP_carrier_limit(n, snakemake)
