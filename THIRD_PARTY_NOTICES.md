# Avis de tiers (third-party notices)

Scaline est sous licence MIT (`LICENSE`). Le paquet contient aussi les éléments suivants, chacun sous sa
propre licence ; les textes sont dans `licenses/`.

## Modèles (dossier `models/`)

Poids convertis en safetensors f16 par `scripts/convert_models.py`, sans autre modification (sommes dans
`models/SHA256SUMS`, origine et somme du fichier d'origine dans l'en-tête de chaque fichier).

| Fichier                            | Modèle                                   | Origine                                  | Licence                                                                                                                                         |
| ---------------------------------- | ---------------------------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `realesr-general-x4v3.safetensors` | Real-ESRGAN, SRVGGNetCompact (photo)     | xinntao/Real-ESRGAN, release v0.2.5.0    | BSD-3-Clause, Copyright (c) 2021, Xintao Wang (`licenses/Real-ESRGAN-LICENSE.txt`)                                                              |
| `realesr-animevideov3.safetensors` | Real-ESRGAN, SRVGGNetCompact (animation) | xinntao/Real-ESRGAN, release v0.2.5.0    | BSD-3-Clause, Copyright (c) 2021, Xintao Wang (`licenses/Real-ESRGAN-LICENSE.txt`)                                                              |
| `rife-4.25.safetensors`            | RIFE 4.25 (IFNet_HDv3)                   | hzwer/Practical-RIFE, v4.25 (2024-09-19) | MIT, Copyright (c) 2021 hzwer ; les poids sont « sous la même licence MIT » d'après le README du projet (`licenses/Practical-RIFE-LICENSE.txt`) |

Le nom des auteurs de Real-ESRGAN n'est pas utilisé pour promouvoir Scaline (clause 3 de la licence BSD).

Jeux de données d'entraînement : Real-ESRGAN est entraîné sur DF2K (DIV2K, « académique seulement », et
Flickr2K) et OST, RIFE sur Vimeo-90K (sans licence publiée). Les licences du code et des poids permettent un
usage commercial ; l'effet des conditions des jeux de données sur les poids n'est tranché par aucun texte.
Zone grise connue de tout le secteur, signalée ici en connaissance de cause.

## Bibliothèques Rust du programme natif

Compilées dans `bin/scaline-worker-*`. Toutes sous licences permissives : MIT, Apache-2.0, Zlib, ISC,
BSD-2-Clause, Unlicense ou Unicode-3.0 (au choix quand plusieurs sont proposées). Textes types : MIT
(`licenses/MIT-wgpu.txt`), Apache-2.0 (`licenses/Apache-2.0.txt`), Zlib (`licenses/Zlib-foldhash.txt`), ISC
(`licenses/ISC-libloading.txt`), Unicode-3.0 (`licenses/Unicode-3.0-unicode-ident.txt`) ; l'avis de copyright de
chaque bibliothèque est dans ses sources publiées sur crates.io.

Liste tirée de `worker/Cargo.lock` (toutes plateformes) :
`cargo metadata --format-version 1 | jq -r '.packages[] | select(.name != "scaline-worker") | "| \(.name) | \(.version) | \(.license) |"' | sort`

| Bibliothèque                         | Version             | Licence                             |
| ------------------------------------ | ------------------- | ----------------------------------- |
| allocator-api2                       | 0.2.21              | MIT OR Apache-2.0                   |
| android_system_properties            | 0.1.6               | MIT OR Apache-2.0                   |
| arrayvec                             | 0.7.8               | MIT OR Apache-2.0                   |
| ash                                  | 0.38.0+1.3.281      | MIT OR Apache-2.0                   |
| autocfg                              | 1.5.1               | Apache-2.0 OR MIT                   |
| bit-set                              | 0.10.0              | Apache-2.0 OR MIT                   |
| bit-vec                              | 0.9.1               | Apache-2.0 OR MIT                   |
| bitflags                             | 2.13.2              | MIT OR Apache-2.0                   |
| block2                               | 0.6.2               | MIT                                 |
| bumpalo                              | 3.20.4              | MIT OR Apache-2.0                   |
| bytemuck                             | 1.25.2              | Zlib OR Apache-2.0 OR MIT           |
| bytemuck_derive                      | 1.12.1              | Zlib OR Apache-2.0 OR MIT           |
| cfg-if                               | 1.0.5               | MIT OR Apache-2.0                   |
| cfg_aliases                          | 0.2.2               | MIT                                 |
| codespan-reporting                   | 0.13.1              | Apache-2.0                          |
| crossbeam-deque                      | 0.8.8               | MIT OR Apache-2.0                   |
| crossbeam-epoch                      | 0.9.21              | MIT OR Apache-2.0                   |
| crossbeam-utils                      | 0.8.23              | MIT OR Apache-2.0                   |
| crunchy                              | 0.2.4               | MIT                                 |
| dispatch2                            | 0.3.1               | Zlib OR Apache-2.0 OR MIT           |
| document-features                    | 0.2.12              | MIT OR Apache-2.0                   |
| either                               | 1.19.0              | MIT OR Apache-2.0                   |
| equivalent                           | 1.0.3               | Apache-2.0 OR MIT                   |
| foldhash                             | 0.2.0               | Zlib                                |
| futures-core                         | 0.3.34              | MIT OR Apache-2.0                   |
| futures-task                         | 0.3.34              | MIT OR Apache-2.0                   |
| futures-util                         | 0.3.34              | MIT OR Apache-2.0                   |
| gpu-allocator                        | 0.28.0              | MIT OR Apache-2.0                   |
| half                                 | 2.7.1               | MIT OR Apache-2.0                   |
| hashbrown                            | 0.16.1              | MIT OR Apache-2.0                   |
| hashbrown                            | 0.17.2              | MIT OR Apache-2.0                   |
| indexmap                             | 2.14.2              | Apache-2.0 OR MIT                   |
| itoa                                 | 1.0.19              | MIT OR Apache-2.0                   |
| js-sys                               | 0.3.107             | MIT OR Apache-2.0                   |
| libc                                 | 0.2.191             | MIT OR Apache-2.0                   |
| libloading                           | 0.8.9               | ISC                                 |
| libm                                 | 0.2.16              | MIT                                 |
| litrs                                | 1.1.0               | MIT OR Apache-2.0                   |
| lock_api                             | 0.4.14              | MIT OR Apache-2.0                   |
| log                                  | 0.4.35              | MIT OR Apache-2.0                   |
| memchr                               | 2.8.3               | Unlicense OR MIT                    |
| naga                                 | 30.0.2              | MIT OR Apache-2.0                   |
| naga-types                           | 30.0.2              | MIT OR Apache-2.0                   |
| num-traits                           | 0.2.19              | MIT OR Apache-2.0                   |
| objc2                                | 0.6.5               | MIT                                 |
| objc2-core-foundation                | 0.3.2               | Zlib OR Apache-2.0 OR MIT           |
| objc2-core-graphics                  | 0.3.2               | Zlib OR Apache-2.0 OR MIT           |
| objc2-encode                         | 4.1.0               | MIT                                 |
| objc2-foundation                     | 0.3.2               | MIT                                 |
| objc2-io-surface                     | 0.3.2               | Zlib OR Apache-2.0 OR MIT           |
| objc2-metal                          | 0.3.2               | Zlib OR Apache-2.0 OR MIT           |
| objc2-quartz-core                    | 0.3.2               | Zlib OR Apache-2.0 OR MIT           |
| once_cell                            | 1.21.4              | MIT OR Apache-2.0                   |
| ordered-float                        | 5.5.1               | MIT                                 |
| parking_lot                          | 0.12.5              | MIT OR Apache-2.0                   |
| parking_lot_core                     | 0.9.12              | MIT OR Apache-2.0                   |
| pin-project-lite                     | 0.2.18              | Apache-2.0 OR MIT                   |
| pollster                             | 1.0.3               | MIT OR Apache-2.0                   |
| portable-atomic                      | 1.15.0              | Apache-2.0 OR MIT                   |
| portable-atomic-util                 | 0.2.8               | Apache-2.0 OR MIT                   |
| presser                              | 0.3.1               | MIT OR Apache-2.0                   |
| proc-macro2                          | 1.0.107             | MIT OR Apache-2.0                   |
| profiling                            | 1.0.18              | MIT OR Apache-2.0                   |
| quote                                | 1.0.47              | MIT OR Apache-2.0                   |
| range-alloc                          | 0.1.5               | MIT OR Apache-2.0                   |
| raw-window-handle                    | 0.6.2               | MIT OR Apache-2.0 OR Zlib           |
| raw-window-metal                     | 1.1.0               | MIT OR Apache-2.0                   |
| rayon                                | 1.12.0              | MIT OR Apache-2.0                   |
| rayon-core                           | 1.13.0              | MIT OR Apache-2.0                   |
| redox_syscall                        | 0.5.18              | MIT                                 |
| renderdoc-sys                        | 1.1.0               | MIT OR Apache-2.0                   |
| rustc-hash                           | 1.1.0               | Apache-2.0/MIT                      |
| rustversion                          | 1.0.23              | MIT OR Apache-2.0                   |
| scopeguard                           | 1.2.0               | MIT OR Apache-2.0                   |
| serde                                | 1.0.229             | MIT OR Apache-2.0                   |
| serde_core                           | 1.0.229             | MIT OR Apache-2.0                   |
| serde_derive                         | 1.0.229             | MIT OR Apache-2.0                   |
| serde_json                           | 1.0.154             | MIT OR Apache-2.0                   |
| slab                                 | 0.4.12              | MIT                                 |
| smallvec                             | 1.16.6              | MIT OR Apache-2.0                   |
| spirv                                | 0.4.0+sdk-1.4.341.0 | Apache-2.0                          |
| static_assertions                    | 1.1.0               | MIT OR Apache-2.0                   |
| syn                                  | 2.0.119             | MIT OR Apache-2.0                   |
| syn                                  | 3.0.10              | MIT OR Apache-2.0                   |
| thiserror                            | 2.0.21              | MIT OR Apache-2.0                   |
| thiserror-impl                       | 2.0.21              | MIT OR Apache-2.0                   |
| tokio                                | 1.54.1              | MIT                                 |
| unicode-ident                        | 1.0.27              | (MIT OR Apache-2.0) AND Unicode-3.0 |
| unicode-width                        | 0.2.3               | MIT OR Apache-2.0                   |
| wasm-bindgen                         | 0.2.130             | MIT OR Apache-2.0                   |
| wasm-bindgen-futures                 | 0.4.80              | MIT OR Apache-2.0                   |
| wasm-bindgen-macro                   | 0.2.130             | MIT OR Apache-2.0                   |
| wasm-bindgen-macro-support           | 0.2.130             | MIT OR Apache-2.0                   |
| wasm-bindgen-shared                  | 0.2.130             | MIT OR Apache-2.0                   |
| web-sys                              | 0.3.107             | MIT OR Apache-2.0                   |
| wgpu                                 | 30.0.2              | MIT OR Apache-2.0                   |
| wgpu-core                            | 30.0.2              | MIT OR Apache-2.0                   |
| wgpu-core-deps-apple                 | 30.0.2              | MIT OR Apache-2.0                   |
| wgpu-core-deps-windows-linux-android | 30.0.2              | MIT OR Apache-2.0                   |
| wgpu-hal                             | 30.0.2              | MIT OR Apache-2.0                   |
| wgpu-naga-bridge                     | 30.0.2              | MIT OR Apache-2.0                   |
| wgpu-types                           | 30.0.2              | MIT OR Apache-2.0                   |
| windows                              | 0.62.2              | MIT OR Apache-2.0                   |
| windows-collections                  | 0.3.2               | MIT OR Apache-2.0                   |
| windows-core                         | 0.62.2              | MIT OR Apache-2.0                   |
| windows-future                       | 0.3.2               | MIT OR Apache-2.0                   |
| windows-implement                    | 0.60.2              | MIT OR Apache-2.0                   |
| windows-interface                    | 0.59.3              | MIT OR Apache-2.0                   |
| windows-link                         | 0.2.1               | MIT OR Apache-2.0                   |
| windows-numerics                     | 0.3.1               | MIT OR Apache-2.0                   |
| windows-result                       | 0.4.1               | MIT OR Apache-2.0                   |
| windows-strings                      | 0.5.1               | MIT OR Apache-2.0                   |
| windows-threading                    | 0.2.1               | MIT OR Apache-2.0                   |
| zerocopy                             | 0.8.66              | BSD-2-Clause OR Apache-2.0 OR MIT   |
| zerocopy-derive                      | 0.8.66              | BSD-2-Clause OR Apache-2.0 OR MIT   |
| zmij                                 | 1.0.23              | MIT                                 |
