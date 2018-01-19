# Image Generation
Compile mcx binary

`mcx_ImageDenoising/data/mcx/src$ make`

Download nightly version of mcxlab. You may overwrite the default one.

`http://mcx.space/nightly/`


go to mcxlab, open matlab

`mcx_ImageDenoising/data/mcx/mcxlab$ matlab &`

Run the following scripts to generate image (in mat format).

* genImage_osa_1e5_to_1e8.m
* genImage_osa_1e9.m (10x1e8 and take average, as the clean image)
