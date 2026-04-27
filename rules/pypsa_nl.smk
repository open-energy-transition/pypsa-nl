# SPDX-FileCopyrightText: Contributors to Open-TYNDP <https://github.com/open-energy-transition/open-tyndp>
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-FileCopyrightText: Open Energy Transition gGmbH
#
# SPDX-License-Identifier: MIT

# PyPSA-NL specific
rule build_tennet_busshapes:
    message:
        "Building the busshape based on TenneT"
    input:
        admin_shapes=resources("admin_shapes.geojson"),
        pocketsWGS="data/tennet/Target Grid Map 2 0 WGS ArcGisOnline - PocketsWGS.geojson"
    output:
        busshape="data/busshapes/base_s_{clusters}_{base_network}.geojson"
    log:
        logs("build_tennet_busshapes_{clusters}_{base_network}.log"),
    script:
        scripts("build_tennet_busshapes.py")


rule create_netherland_core:
    message:
        "Build an unsolved, high-resolution, electricity-only model for the Netherlands"
    output:
        network=resources("elec_nl.nc"),
    shell:
        """
        snakemake resources/nl-core/networks/base_s_21_elec.nc --configfile config/config.nl-core.yaml -call --unlock
        sleep 5
        snakemake resources/nl-core/networks/base_s_21_elec.nc --configfile config/config.nl-core.yaml -call
        mv resources/nl-core/networks/base_s_21_elec.nc {output}
        """

rule final_adjustment_myopic:
    message:
        "Combine two networks"
    input:
        network_nl=resources("elec_nl.nc"),
        network=resources("networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_brownfield.nc"),
    output:
        network=resources(
            "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_brownfield_adjusted.nc"
        ),
    script:
        scripts("final_adjustment_myopic.py")