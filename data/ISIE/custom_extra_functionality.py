# SPDX-FileCopyrightText: : 2023- The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

def add_battery_duration_fix(n, snakemake):
    """
    Add constraint ensuring that P/E ratio for battery in NL is proportional to max hours
    """

    bus_country = n.buses.index[n.buses.country.isin(["NL"])]

    energy_bool = n.stores.carrier.isin(["battery"]) & n.stores.bus.isin(bus_country)
    energy_ext = n.stores[energy_bool].query("e_nom_extendable")

    charger_bool = n.links.carrier.isin(["battery charger"]) & n.links.bus0.isin(bus_country)
    chargers_ext = n.links[charger_bool].query("p_nom_extendable")

    if not energy_ext.empty and not chargers_ext.empty:
        expr = (
            n.model["Store-e_nom"].loc[energy_ext.index]
            == n.model["Link-p_nom"].loc[chargers_ext.index]
            * snakemake.config["electricity"]["max_hours"]["battery"]
        )

        n.model.add_constraints(expr, name="NL-Battery-duration")


def distribute_TYNDP_carrier_limit(n, snakemake):
    expandable = snakemake.config["tennet_adjustment"]["extendable_carriers"]["Generator"]

    # For Links
    link_p_nom_bool = ~n.links.p_nom_group.isna() & ~n.links.carrier.isin(expandable)
    link_p_nom = n.links[link_p_nom_bool].query("p_nom_extendable")

    if not link_p_nom.empty:
        expr_1 = (
            n.model["Link-p_nom"].loc[link_p_nom.index].groupby(link_p_nom.carrier).sum() 
            == link_p_nom.groupby("carrier")["p_nom_group"].mean()
        )

        n.model.add_constraints(expr_1, name="Link-dist-TYNDP")

    link_p_min_bool = ~n.links.p_nom_group.isna() & n.links.carrier.isin(expandable)
    link_p_min = n.links[link_p_min_bool].query("p_nom_extendable")

    if not link_p_min.empty:
        expr_2 = (
            n.model["Link-p_nom"].loc[link_p_min.index].groupby(link_p_min.carrier).sum() 
            >= link_p_min.groupby("carrier")["p_nom_group"].mean()
        )

        n.model.add_constraints(expr_2, name="Link-min-TYNDP")

    # For Generators
    gen_p_nom_bool = ~n.generators.p_nom_group.isna() & ~n.generators.carrier.isin(expandable)
    gen_p_nom = n.generators[gen_p_nom_bool].query("p_nom_extendable")

    if not gen_p_nom.empty:
        expr_3 = (
            n.model["Generator-p_nom"].loc[gen_p_nom.index].groupby(gen_p_nom.carrier).sum() 
            == gen_p_nom.groupby("carrier")["p_nom_group"].mean()
        )

        n.model.add_constraints(expr_3, name="Generator-dist-TYNDP")

    gen_p_min_bool = ~n.generators.p_nom_group.isna() & n.generators.carrier.isin(expandable)
    gen_p_min = n.generators[gen_p_min_bool].query("p_nom_extendable")

    if not gen_p_min.empty:
        expr_4 = (
            n.model["Generator-p_nom"].loc[gen_p_min.index].groupby(gen_p_min.carrier).sum() 
            >= gen_p_min.groupby("carrier")["p_nom_group"].mean()
        )

        n.model.add_constraints(expr_4, name="Generator-min-TYNDP")


def custom_extra_functionality(n, snapshots, snakemake):

    add_battery_duration_fix(n, snakemake)

    distribute_TYNDP_carrier_limit(n, snakemake)