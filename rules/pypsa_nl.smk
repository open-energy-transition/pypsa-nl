# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-FileCopyrightText: Open Energy Transition gGmbH
#
# SPDX-License-Identifier: MIT


configfile: "config/ISIE/pypsa-nl.yaml"


# PyPSA-NL specific
rule build_tennet_busshapes:
    message:
        "Building the busshape based on TenneT"
    input:
        admin_shapes=resources("admin_shapes.geojson"),
        archetypen_buurten="data/ISIE/archetypen_buurten.geojson",
        pockets_traces="data/ISIE/pockets_traces.geojson",
    output:
        pockets_archetypes=resources("pockets_archetypes_{clusters}_{base_network}.geojson"),
        busshape="data/busshapes/base_s_{clusters}_{base_network}.geojson",
    # log:
    #     logs("build_tennet_busshapes_{clusters}_{base_network}.log"),
    script:
        scripts("build_tennet_busshapes.py")


def align_configuration(w):
    import yaml

    config_temp = {
        "scenario": {"planning_horizons": [int(w.planning_horizons)]},
        "snapshots": config["snapshots"],
        "atlite": config["atlite"],
        "clustering": {"temporal": config["clustering"]["temporal"]},
    }

    with open("config.temp.yaml", "w") as file:
        yaml.safe_dump(config_temp, file, default_flow_style=False)

    return []


rule create_netherland_core:
    message:
        "Build an unsolved, high-resolution, electricity-only model for the Netherlands"
    input:
        check=align_configuration,
        config="config/config.nl-core.yaml",
    params:
        resolution_sector=config_provider("clustering", "temporal", "resolution_sector"),
    output:
        network=resources("nl_{planning_horizons}.nc"),
    shell:
        """
        snakemake resources/nl-core/networks/base_s_21___{wildcards.planning_horizons}_brownfield.nc --configfile config/ISIE/config.nl-core.yaml config.temp.yaml --rerun-incomplete -call
        mv resources/nl-core/networks/base_s_21___{wildcards.planning_horizons}_brownfield.nc {output}
        rm config.temp.yaml
        """


rule final_adjustment_myopic:
    message:
        "Combine two networks"
    params:
        offshore_buses=config_provider("offshore_buses"),
        res_tyndp_mapping=config_provider("res_tyndp_mapping"),
        conv_tyndp_mapping=config_provider("conv_tyndp_mapping"),
        store_tyndp_mapping=config_provider("store_tyndp_mapping"),
        tennet_capacity=config_provider("tennet_capacity"),
    input:
        network_nl=resources("nl_{planning_horizons}.nc"),
        network=resources(
            "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_brownfield.nc"
        ),
    output:
        network=resources(
            "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_brownfield_adjusted.nc"
        ),
    script:
        scripts("final_adjustment_myopic.py")
