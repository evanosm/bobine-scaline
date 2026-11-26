# Scaline

Plugin maison de [Bobine](https://genez.app/bobine) : agrandit un clip vidéo en 4K (Real-ESRGAN) ou en fait un
ralenti fluide en ajoutant des images intermédiaires (RIFE). Le calcul tourne sur le GPU de la machine, sans réseau.
Le résultat est un nouveau média, proposé avant d'être appliqué ; l'original reste dans le projet.

Licence MIT. Gratuit. Bobine 0.4.0 ou plus (`media.frames` avec `every_frame`, `media.encode`, `ReplaceClipMedia`).

## Ce que voit l'utilisateur

Deux commandes, dans le menu du clip et le menu Plugins. Elles prennent le clip vidéo sélectionné, sinon celui
sous la tête de lecture.

- **Agrandir en 4K** : chaque image du clip est agrandie ×4 par Real-ESRGAN, puis ramenée à la taille visée : celle
  de la séquence si elle est plus grande que le clip (l'image y tient entière), sinon 3840 de large (réglable).
  Une séquence 4K reçoit donc un clip 1080p en 3840 × 2160 ; un clip 720p dans une séquence 1080p passe en
  1920 × 1080.
- **Ralenti fluide (images intermédiaires)** : RIFE calcule une image entre chaque paire d'images ; le nouveau
  média a deux fois plus d'images à cadence double (30 → 60 im./s), puis le clip passe à la moitié de sa vitesse
  (`SetClipSpeed`, ripple : les clips suivants se décalent). Le son lié est ralenti avec lui (règle de Bobine pour
  les clips liés). Réglage « garder la vitesse » : seulement la cadence doublée, même durée (utile dans une
  séquence à 60 im./s).

Avant de lancer, une notification annonce ce qui va être fait et la durée estimée, mesurée sur le GPU de la
machine (« Agrandissement 1920 × 1080 → 3840 × 2160 (photo) : 240 images, environ 15 min sur Apple M1 Pro »).
La progression s'affiche dans la barre des tâches de fond. À la fin, une proposition « Remplacer le média du
clip » (plus la vitesse pour le ralenti) : Appliquer ou Annuler. Le clip garde ses propriétés, images clés,
transitions et son lien.

Choix de conception : deux commandes, une par intention (plus net, 4K ou ralenti), et un seul réglage pour le cas
moins courant (cadence doublée sans ralentir) plutôt qu'une troisième commande.

Réglages (« Mes extensions ») :

| Réglage                            | Par défaut | Effet                                                                                 |
| ---------------------------------- | ---------- | ------------------------------------------------------------------------------------- |
| Modèle : `photo` ou `animation`    | `photo`    | `realesr-general-x4v3` (prise de vue réelle) ou `realesr-animevideov3` (dessin animé) |
| Largeur visée                      | 3840       | Taille de sortie quand la séquence n'est pas plus grande que le clip                  |
| Ralenti fluide : garder la vitesse | non        | Cadence doublée seulement, sans `SetClipSpeed`                                        |

## Comment ça marche

```
plugin (iframe)                         Bobine                          programme natif (Rust, wgpu)
project.get → clip, média
native.call("estimate")  ────────────────────────────────────────────▶  mesure 1 image sur ce GPU
media.encode.begin       ──────────────▶ session (ProRes .mov)
pour chaque lot (≤ 384 Mo d'images) :
  media.frames(every_frame, rgba) ─────▶ frames-<n>.raw/json
  native.call("upscale" | "interpolate") ─────────────────────────────▶ scaline-<n>.raw (RGBA)
  media.encode.append(raw) ────────────▶ encode, supprime raw
media.encode.finish      ──────────────▶ import → { media }
project.propose(ReplaceClipMedia { clip, media, source_in }, [SetClipSpeed])
```

- Lots : la plage du média lue par le clip (vitesse comprise), élargie d'une image de chaque côté. Chaque lot
  reprend la dernière image du précédent : elle sert d'image précédente à RIFE, et le programme natif ne la rend
  pas deux fois (`after` : les images d'instant ≤ `after` ne sont pas réécrites).
- Appels au programme natif : ~45 s de calcul et 256 Mo écrits au plus chacun (le délai de Bobine est 10 min).
- `source_in` du clip dans le nouveau média = `source_in` d'origine − instant de la première image rendue.
- Images reçues réduites à 1920 de côté au plus (plafond de `media.frames`) : un clip 4K ou plus n'est pas
  agrandi davantage (« déjà à cette taille »).

### Programme natif (`worker/`)

Protocole 1 des programmes natifs de Bobine (lignes JSON, `--bobine-worker 1`, seul `BOBINE_WORK_DIR` dans
l'environnement). Méthodes `ping`, `info`, `estimate`, `upscale`, `interpolate` (détail dans
`worker/src/protocol.rs`). Les modèles sont lus dans `models/` du paquet, à côté de `bin/`.

L'inférence est écrite en shaders WGSL et tourne sur wgpu : Metal (macOS), DirectX 12 (Windows), Vulkan (Linux).
Sans GPU matériel, wgpu prend le GPU logiciel du système (WARP sous Windows, lavapipe sous Linux s'il est
installé), beaucoup plus lent et annoncé dans la notification ; sans rien, erreur claire (`no_gpu`). Pas d'ONNX
Runtime : quatre `ort` statiques dépasseraient le plafond de 50 Mo d'un paquet.

Opérations (`worker/src/shaders/`) : convolution 3×3 (pas 1 et 2, résidu, PReLU / LeakyReLU), déconvolution 4×4
pas 2, pixel shuffle (plus l'image agrandie au plus proche de Real-ESRGAN), redimensionnement bilinéaire, warp
(`grid_sample` bilinéaire, bord répété), sigmoïde et mélange, concaténation, addition, passage RGBA 8 bits ↔
tenseur. Tenseurs rangés par plans de 4 canaux, stockés en f16 si l'adaptateur le permet (`SHADER_F16`), sinon
en f32.

- Real-ESRGAN (SRVGGNetCompact) : 34 convolutions 64 canaux, pixel shuffle ×4. Image découpée en tuiles de
  512 px au plus, lues avec une marge de 34 px (une par convolution, le champ de vision du réseau) : le centre de
  chaque tuile est identique au calcul sur l'image entière, sans couture, et la mémoire du GPU reste bornée
  (~45 Mo par tenseur). Calculs en f16 quand le GPU le permet (deux fois plus rapide), comme l'inférence
  d'origine de Real-ESRGAN (`half=True`). Sortie ×4 ramenée à la taille visée sur le processeur (moyenne de
  surface pour réduire, bilinéaire pour agrandir).
- RIFE 4.25 (IFNet_HDv3, 5 blocs de 1/16 à 1/1) : image entière complétée de zéros jusqu'à un multiple de 128,
  comme `inference_video.py` ; `timestep` 0,5, `scale` 1. Calculs en f32 (stockage f16) : en f16, le flux
  s'écartait de quelques niveaux par endroits pour un gain faible.

## Vérification

Sortie des shaders comparée à PyTorch 2.15 (f32, processeur) sur des images d'essai (`scripts/reference.py`
écrit les entrées et les références, `cargo run --release --example compare -- <dossier> ../models` compare).
Écarts en niveaux sur 255, images 8 bits, Apple M1 Pro (Metal), le 2026-11-26 :

| Réseau (image d'essai)                       | Stockage / calculs     | vs PyTorch, mêmes poids f16 | vs PyTorch, poids f32 d'origine |
| -------------------------------------------- | ---------------------- | --------------------------- | ------------------------------- |
| Real-ESRGAN photo (96 × 64 → 384 × 256)      | f32 / f32              | max 1, 100 % exacts         | max 1, 98,2 % exacts            |
| Real-ESRGAN photo                            | f16 / f16 (par défaut) | max 4, moyenne 0,24         | max 4, moyenne 0,24             |
| Real-ESRGAN animation                        | f32 / f32              | max 1, 100 % exacts         | max 1, 98,5 % exacts            |
| Real-ESRGAN animation                        | f16 / f16 (par défaut) | max 3, moyenne 0,16         | max 3, moyenne 0,16             |
| RIFE 4.25 (200 × 120, complétée à 256 × 128) | f32 / f32              | max 1, 99,99 % exacts       | max 5, moyenne 0,009            |
| RIFE 4.25                                    | f16 / f32 (par défaut) | max 6, moyenne 0,05         | max 2, moyenne 0,05             |

Les écarts « max 1 » viennent de l'arrondi final (une valeur à x,5 près). Les tests `worker/tests/ops.rs`
comparent aussi chaque opération à un calcul de référence (écart < 1e-4 en f32) ; la CI les fait tourner sur
lavapipe, le Vulkan logiciel de Mesa.

Vitesse, même Mac (M1 Pro, 16 cœurs GPU), image 1920 × 1080 :

| Calcul                                            | Temps par image                        |
| ------------------------------------------------- | -------------------------------------- |
| Agrandir en 4K, photo (1920 × 1080 → 3840 × 2160) | 3,8 s (3,7 s GPU + 0,1 s de réduction) |
| Agrandir en 4K, animation                         | 1,8 s                                  |
| Image intermédiaire RIFE                          | 0,21 s                                 |

Soit environ 19 min pour 10 s de vidéo à 30 im./s en photo, 1 min pour en faire un ralenti fluide.

## Modèles

Poids convertis en safetensors f16 (`scripts/convert_models.py`, noms de tenseurs d'origine), publiés dans la
release [`modeles-1`](https://github.com/evanosm/bobine-scaline/releases/tag/modeles-1) de ce dépôt, sommes dans
`models/SHA256SUMS` ; les poids d'origine ne sont pas dans git.

| Fichier                            | Taille | Origine                                                                                                            | SHA-256 de l'original |
| ---------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------ | --------------------- |
| `realesr-general-x4v3.safetensors` | 2,4 Mo | [Real-ESRGAN v0.2.5.0](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth) | `8dc7edb9…6292`       |
| `realesr-animevideov3.safetensors` | 1,2 Mo | [Real-ESRGAN v0.2.5.0](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth) | `b8a83768…b75d`       |
| `rife-4.25.safetensors`            | 11 Mo  | [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) 4.25 (Google Drive, `flownet.pkl` de l'archive)          | `6615790e…6de2`       |

RIFE 4.25 est la version conseillée par Practical-RIFE « pour la plupart des scènes » ; ses poids se téléchargent.

Licences : Real-ESRGAN BSD-3-Clause, Practical-RIFE MIT (poids « sous la même licence » d'après son README). Textes
dans `licenses/`, détail et bibliothèques Rust dans `THIRD_PARTY_NOTICES.md`, tous deux dans le paquet.
Zone grise connue : les jeux d'entraînement (DF2K, dont DIV2K « académique seulement », pour Real-ESRGAN ;
Vimeo-90K, sans licence publiée, pour RIFE). Les licences du code et des poids permettent l'usage commercial ;
l'effet des conditions des jeux de données sur les poids n'est tranché par aucun texte.

Refaire les modèles (torch dans un environnement Python à part) :

```sh
python scripts/convert_models.py <dossier des poids d'origine> models/
python scripts/reference.py <dossier des poids d'origine> models/ /tmp/scaline-ref
cd worker && cargo run --release --example compare -- /tmp/scaline-ref ../models
```

## Développer

```sh
scripts/fetch-models.sh   # modèles (release modeles-1), sommes vérifiées
pnpm install              # SDK pris dans ../bobine-plugin-sdk
pnpm lint && pnpm typecheck && pnpm test
cd worker && cargo test   # GPU requis pour les calculs (sautés sinon)
CARGO_TARGET_DIR=/tmp/scaline-target pnpm build   # package/ : JS, modèles, programme de cette machine
pnpm check                # mêmes règles que Bobine
```

Ne jamais compiler dans `package/` : un `target/` y dépasse les plafonds d'un paquet. Essai dans Bobine : source
« dev » pointant sur `package/` (`extensions.json`), Bobine 0.4.0 ou plus.

CI (`.github/workflows/ci.yml`, à chaque push, Ubuntu seulement) : clippy, tests du programme natif sur lavapipe,
Biome, typecheck, tests du plugin, `bobine-plugin check`. Au tag `v<version>` (`release.yml`) : programme compilé
et testé sur les quatre plateformes (macOS arm64 et x64, Linux x64 glibc, Windows x64 sur WARP), un paquet
`.bobine-ext` (< 50 Mo, vérifié) joint à la release. Publication sur la marketplace : manuelle, compte admin.

## Limites connues

- Pas de son dans le média rendu : le clip son lié garde l'original (règle de `ReplaceClipMedia`).
- Cadence constante supposée pour placer les images intermédiaires (`fps` nominal du média).
- Un seul rendu à la fois (une session `media.encode` par plugin).
- Le ralenti double la durée du clip et décale les clips suivants (ripple).
- macOS Intel, Linux et Windows : compilés et testés en CI (lavapipe, WARP), pas essayés sur une vraie machine.
