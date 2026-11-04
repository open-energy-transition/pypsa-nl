# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-FileCopyrightText: Open Energy Transition gGmbH
#
# SPDX-License-Identifier: MIT


configfile: "config/ISIE/pypsa-nl.yaml"


# PyPSA-NL specific
rule build_tennet_busshapes:
    input:
        admin_shapes=resources("admin_shapes.geojson"),
        archetypen_buurten="data/ISIE/archetypen_buurten.geojson",
        pockets_traces="data/ISIE/pockets_traces.geojson",
    output:
        # pockets_archetypes=resources("pockets_archetypes_{clusters}_{base_network}.geojson"),
        busshape="data/busshapes/simplified_{clusters}_osm.geojson",
    message:
        "Building the busshape based on TenneT"
    # log:
    #     logs("build_tennet_busshapes_{clusters}_{base_network}.log"),
    script:
        scripts("build_tennet_busshapes.py")


def align_configuration(w):
    import yaml

    config_temp = {
        "planning_horizons": [int(w.horizon)],
        "snapshots": config["snapshots"],
        "atlite": config["atlite"],
        "clustering": {"temporal": config["clustering"]["temporal"]},
    }

    with open("config.temp.yaml", "w") as file:
        yaml.safe_dump(config_temp, file, default_flow_style=False)

    return []


rule create_netherland_core:
    input:
        check=align_configuration,
        config="config/ISIE/config.nl-core.yaml",
    output:
        network=resources("nl_{horizon}.nc"),
    params:
        resolution_sector=config_provider("clustering", "temporal", "resolution_sector"),
    message:
        "Build an unsolved, high-resolution, electricity-only model for the Netherlands"
    shell:
        """
        snakemake resources/nl-core/networks/composed_{wildcards.horizon}.nc --configfile config/ISIE/config.nl-core.yaml config.temp.yaml --rerun-incomplete -call --unlock
        snakemake resources/nl-core/networks/composed_{wildcards.horizon}.nc --configfile config/ISIE/config.nl-core.yaml config.temp.yaml --rerun-incomplete -call
        mv resources/nl-core/networks/composed_{wildcards.horizon}.nc {output}
        rm config.temp.yaml
        """


rule final_adjustment_myopic:
    input:
        network_nl=resources("nl_{horizon}.nc"),
        network=resources("networks/composed_{horizon}.nc"),
    output:
        network=resources("networks/composed_{horizon}_adjusted.nc"),
    params:
        tennet_adjustment=config_provider("tennet_adjustment"),
        offshore_buses=config_provider("offshore_buses"),
        # res_tennet_mapping=config_provider("res_tennet_mapping"),
        res_tyndp_mapping=config_provider("res_tyndp_mapping"),
        store_tyndp_mapping=config_provider("store_tyndp_mapping"),
    message:
        "Combine two networks"
    script:
        scripts("final_adjustment_myopic.py")
