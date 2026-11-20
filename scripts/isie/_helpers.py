# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-FileCopyrightText: Open Energy Transition gGmbH
#
# SPDX-License-Identifier: MIT
import pypsa

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


def make_renewables_expandable(
    n: pypsa.Network,
    res_carrier: list[str],
    min_ext_factor: float,
    countries: list[str] = ["NL"],
) -> pypsa.Network:
    """
    Make renewable generators in the given countries extendable, moving their
    existing capacity per carrier into ``p_nom_group``.

    Parameters
    ----------
    n : pypsa.Network
        Network whose generators are modified.
    res_carrier : list[str]
        Renewable carriers to make extendable.
    min_ext_factor : float


    Returns
    -------
    pypsa.Network
        The modified network.
    """

    gens = n.generators

    # Netherland Focus
    buses_idx_nl = n.buses.index[n.buses.country.isin(countries)]
    res_idx_nl = gens.index[
        gens.carrier.isin(res_carrier) & gens.bus.isin(buses_idx_nl)
    ]

    # use p_nom_group to distribute capacity of TYNDP technologies
    gens.loc[res_idx_nl, "p_nom_extendable"] = True
    gens.loc[res_idx_nl, "p_nom_group"] = (
        gens.loc[res_idx_nl].groupby("carrier")["p_nom"].transform("sum")
    )
    gens.loc[res_idx_nl, "p_nom"] = 0
    gens.loc[res_idx_nl, "p_nom_min"] = 0

    # Non Netherland Countries
    buses_idx = n.buses.index[~n.buses.country.isin(countries)]
    res_idx = gens.index[gens.carrier.isin(res_carrier) & gens.bus.isin(buses_idx)]

    # use p_nom_group to distribute capacity of TYNDP technologies
    gens.loc[res_idx, "p_nom_extendable"] = True

    gens.loc[res_idx, "p_nom_min"] *= min_ext_factor

    return n


def make_all_battery_extendable(n):

    bus_country = n.buses.index[~n.buses.country.isin(["NL"])]
    stores_i = n.stores.index[
        (n.stores.carrier == "battery") & n.stores.bus.isin(bus_country)
    ]
    links_i = n.links.index[
        ((n.links.carrier == "battery charger") & n.links.bus0.isin(bus_country))
        | ((n.links.carrier == "battery discharger") & n.links.bus1.isin(bus_country))
    ]

    n.stores.loc[stores_i, "e_nom_extendable"] = True
    n.links.loc[links_i, "p_nom_extendable"] = True

    return n
